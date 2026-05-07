<font color="#FF5722">C++端（注意yolov5.cpp文件为yolo结构源码）</font>

一、从torch权重文件生成yolov5sv3.wts文件

克隆两个项目仓库得到模型文件yolov5sv3.pt

git clone https://github.com/hlld/tensorrt-yolov5.git

git clone https://github.com/ultralytics/yolov5.git

将tensorrt-yolov5目录下的gen_wts.py脚本，复制到ultralytics/yolov5项目根目录（确保gen_wts.py脚本里配置的权重名、生成文件名都是yolov5sv3.pt和yolov5sv3.wts）

进入ultralytics/yolov5目录，执行命令生成wts文件（python gen_wts.py）

运行完成后会在当前目录生成yolov5sv3.wts文件

二、编译构建tensorrt-yolov5得到引擎文件并运行推理

把上一步生成的yolov5sv3.wts放到tensorrt-yolov5/yolov5目录下

进入tensorrt-yolov5/yolov5目录，创建编译目录并编译

mkdir build

cd build

cmake ..

make

sudo ./yolov5 -s s

执行后会生成yolov5sv3.engine（TensorRT引擎文件）

加载引擎文件，对图片进行推理检测

sudo ./yolov5 -e s -d  ../images

会自动处理images文件夹里的图片  






<font color="#FF5722">Python端<font>

Python加载TensorRT模型运行推理

提前安装python-tensorrt、pycuda等依赖库

确认已经编译生成好 yolov5sv3.engine和libmyplugins.so文件（执行make编译后，附加在build文件夹里）

直接运行Python推理脚本

python yolov5_trt.py

即可通过Python调用TensorRT模型做推理



![image](result_truth(1).jpg)
