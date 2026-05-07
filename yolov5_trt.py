"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import random
import sys
import threading
import time

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision


INPUT_W = 608
INPUT_H = 608
CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


class YoLov5TRT(object):
    def __init__(self, engine_file_path):
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(
                binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def infer(self, input_image_path):
        threading.Thread.__init__(self)
        # 将当前实例设为激活上下文 压入上下文栈顶部
        self.cfx.push()
        # 恢复变量
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # 图像预处理
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(
            input_image_path)
        # 将输入图像数据复制到CPU缓冲区
        np.copyto(host_inputs[0], input_image.ravel())
        # 将输入数据传输到GPU
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # 执行推理
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # 将预测结果从GPU传回CPU
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # 流同步
        stream.synchronize()
        # 移除上下文栈顶部的上下文
        self.cfx.pop()
        # 批次大小为1，直接使用第一行输出
        output = host_outputs[0]
        # 后处理
        result_boxes, result_scores, result_classid = self.post_process(
            output, origin_h, origin_w)
        # 在原始图像上绘制检测框和标签
        for i in range(len(result_boxes)):
            box = result_boxes[i]
            plot_one_box(box, image_raw, label="{}:{:.2f}".format(
                categories[int(result_classid[i])], result_scores[i]))
        parent, filename = os.path.split(input_image_path)
        save_name = os.path.join(parent, "output_"+filename)
        # 保存结果图像
        cv2.imwrite(save_name, image_raw)

    def destory(self):
        self.cfx.pop()

    def preprocess_image(self, input_image_path):
        image_raw = cv2.imread(input_image_path)
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # 计算宽高缩放比例与填充尺寸
        r_w = INPUT_W / w
        r_h = INPUT_H / h
        if r_h > r_w:
            tw = INPUT_W
            th = int(r_w * h)
            tx = 0
            ty = int((INPUT_H - th) / 2)
        else:
            tw = int(r_h * w)
            th = INPUT_H
            tx = int((INPUT_W - tw) / 2)
            ty = 0
        # 保持长宽比例缩放图像
        image = cv2.resize(image, (tw, th))
        # 短边使用(128,128,128)颜色进行填充
        image = cv2.copyMakeBorder(
            image, ty, ty, tx, tx, cv2.BORDER_CONSTANT, (128, 128, 128))
        image = image.astype(np.float32)
        # 归一化到 [0,1] 区间
        image /= 255.0
        # 格式从 HWC 转换为 CHW
        image = np.transpose(image, [2, 0, 1])
        # 格式从 CHW 转换为 NCHW
        image = np.expand_dims(image, axis=0)
        # 转换为连续的内存数组
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        y = torch.zeros_like(x) if isinstance(
            x, torch.Tensor) else np.zeros_like(x)
        r_w = INPUT_W / origin_w
        r_h = INPUT_H / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2]/2
            y[:, 2] = x[:, 0] + x[:, 2]/2
            y[:, 1] = x[:, 1] - x[:, 3]/2 - (INPUT_H - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3]/2 - (INPUT_H - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2]/2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2]/2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3]/2
            y[:, 3] = x[:, 1] + x[:, 3]/2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        # 获取检测到的目标框数量
        num = int(output[0])
        # 重塑为二维数组
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # 转换为PyTorch张量
        pred = torch.Tensor(pred).cuda()
        # 获取检测框坐标
        boxes = pred[:, :4]
        # 获取置信度分数
        scores = pred[:, 4]
        # 获取类别编号
        classid = pred[:, 5]
        # 筛选置信度大于阈值的框
        si = scores > CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        # 执行非极大值抑制NMS
        indices = torchvision.ops.nms(
            boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
        result_boxes = boxes[indices, :].cpu()
        result_scores = scores[indices].cpu()
        result_classid = classid[indices].cpu()
        return result_boxes, result_scores, result_classid


class myThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


if __name__ == '__main__':
    PLUGIN_LIBRARY = 'build/libmyplugins.so'
    ctypes.CDLL(PLUGIN_LIBRARY)
    engine_file_path = "build/yolov5s.engine"

    coco_labels = "coco_labels.txt"
    categories = []
    with open(coco_labels, "r") as f:
        for line in f:
            categories.append(line.strip())

    # YoLov5TRT instance
    yolov5_warpper = YoLov5TRT(engine_file_path)

    input_image_paths = ["zidane.jpg", "bus.jpg"]

    for input_image_path in input_image_paths:
        # create a new thread 
        thread1 = myThread(yolov5_warpper.infer, [input_image_path])
        thread1.start()
        thread1.join()

    # destory the instance
    yolov5_warpper.destory()

