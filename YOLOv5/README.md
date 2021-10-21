# YOLOv5

The Pytorch implementation is [ultralytics/yolov5](https://github.com/ultralytics/yolov5).  Wang-xinyu use TensorRT C++ API to define yolov5 network, build engine and infer in [wang-xinyu/tensorrtx/yolov5](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5).

In this case, I define YOLOv5 network and build engine by TensorRT Python API. If you don't care the effect of the performance on language itself, you can also use the python script provided by me. Otherwise you can refer to `yolov5.cpp` in [wang-xinyu/tensorrtx/yolov5](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5)

## Different versions of yolov5

Currently, we support yolov5 v1.0(yolov5s only), v2.0, v3.0, v3.1, v4.0 and v5.0.

- For yolov5 v5.0, download .pt from [yolov5 release v5.0](https://github.com/ultralytics/yolov5/releases/tag/v5.0), `git clone -b v5.0 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v5.0 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v5.0](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v5.0/yolov5).
- For yolov5 v4.0, download .pt from [yolov5 release v4.0](https://github.com/ultralytics/yolov5/releases/tag/v4.0), `git clone -b v4.0 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v4.0 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v4.0](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v4.0/yolov5).
- For yolov5 v3.1, download .pt from [yolov5 release v3.1](https://github.com/ultralytics/yolov5/releases/tag/v3.1), `git clone -b v3.1 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v3.1 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v3.1](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v3.1/yolov5).
- For yolov5 v3.0, download .pt from [yolov5 release v3.0](https://github.com/ultralytics/yolov5/releases/tag/v3.0), `git clone -b v3.0 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v3.0 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v3.0](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v3.0/yolov5).
- For yolov5 v2.0, download .pt from [yolov5 release v2.0](https://github.com/ultralytics/yolov5/releases/tag/v2.0), `git clone -b v2.0 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v2.0 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v2.0](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v2.0/yolov5).
- For yolov5 v1.0, download .pt from [yolov5 release v1.0](https://github.com/ultralytics/yolov5/releases/tag/v1.0), `git clone -b v1.0 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v1.0 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v1.0](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v1.0/yolov5).

## Config

- Choose the model s/m/l/x/s6/m6/l6/x6 from command line arguments.
- Input shape defined in `yololayer.py`
- Number of classes defined in `yololayer.py`, **DO NOT FORGET TO ADAPT THIS, If using your own model**
- INT8/FP16/FP32 can be selected by the macro in `yolov5.py`, **INT8 need more steps, pls follow `How to Run` first and then go the `INT8 Quantization` below**
- GPU id can be selected by `__init__()` of class `Yolov5TRT` in `yolov5_trt.py`
- NMS thresh in `yolov5_trt.py`
- BBox confidence thresh in `yolov5_trt.py`
- Batch size in `__init__()` of class `Yolov5TRT` in `yolov5_trt.py`

## How to Run, yolov5s as example

1. generate .wts from pytorch with .pt, or download .wts from model zoo.(.wts has been provided in this repo so you can skip this step if you use yolov5s)

   ```shell
   git clone -b v5.0 https://github.com/ultralytics/yolov5.git
   git clone https://github.com/wang-xinyu/tensorrtx.git
   // download https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
   cp {TRTx}/yolov5/gen_wts.py {ultralytics}/yolov5
   cd {ultralytics}/yolov5
   python gen_wts.py -w yolov5s.pt -o yolov5s.wts
   // a file 'yolov5s.wts' will be generated.
   ```
   
2. build `TRTx/yolov5` and run

   ```shell
   cd {tensorrtx}/yolov5/
   // update CLASS_NUM in yololayer.h if your model is trained on custom dataset
   mkdir build
   cd build
   cp {TRTx}/yolov5/yolov5s.wts {tensorrtx}/yolov5/build
   cmake ..
   make //generate 'libmyplugins.so'
   cp {tensorrtx}/yolov5/build/libmyplugins.so {TRTx}/yolov5 //. has been provided in this repo so you can skip this step
   cd {TRTx}/yolov5/
   python yolov5.py // serialize model to plan file, you can change the arguments values of main()
   python yolov5_trt.py // deserialize and run inference, the images in [image folder] will be processed.
   ```

3. check the images generated, as follows: `zidane.jpg` and `bus.jpg`

## INT8 Quantization

TO DO.

# Referring

[yolov5 in wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5)
