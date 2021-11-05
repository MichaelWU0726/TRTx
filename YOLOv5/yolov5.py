import ctypes
from yololayer import *
from Utils.common import GiB

USE_FP16=False
USE_INT8=False
TRT_LOGGER = trt.Logger(trt.Logger.WARNNING)
# Force init TensorRT plugins
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
input_name = 'data'
out_name = 'prob'

# Calculate the width and depth of CSP
# For yolov5s, gw,gd=0.50,0.33, so get_width(128,gw) is 64,get_depth(3,gd) is 1
def get_width(x: int, gw: float, divisor: int=8 ):
    """
    Using gw to control the number of kernels that must be multiples of 8.
    return math.ceil(x / divisor) * divisor
    """
    if x*gw % divisor == 0:
        return int(x*gw)
    return (int(x*gw/divisor)+1)*divisor

def get_depth(x: int, gd: float):
    if x==1:
        return 1
    else:
        return round(x*gd) if round(x*gd) > 1 else 1


def yolov5(network, wts_name):
    # Configure the network layers based on the weights provided. In this case, the weights are imported from a pytorch model.
    # Add an input layer. The name is a string, dtype is a TensorRT dtype, and the shape can be provided as either a list or tuple.
    data = network.add_input(name=input_name, dtype=Yolo.DTYPE, shape=trt.Dims3(3, Yolo.INPUT_H, Yolo.INPUT_W))
    assert data, "Add input failed"

    weights = loadWeights(wts_name)

    #----------backbone------#
    GW = Yolo.GW
    GD = Yolo.GD
    CLASS_NUM = Yolo.CLASS_NUM
    # out_name = Yolo.OUTPUT_ARGS
    # get_width, calculate the number of conv kernels
    focus0 = focus(network, weights, data, 3, get_width(64, GW), 3, "model.0")
    # CBL
    width_128 = get_width(128, GW)  # =64
    depth_3 = get_depth(3, GD)  # =1
    conv1 = convBlock(network, weights, focus0.get_output(0), width_128, 3, 2, 1,"model.1")
    # CSP1_1
    c3_2 = C3(network, weights, conv1.get_output(0), width_128, width_128, depth_3, True, 1, 0.5, "model.2")
    # CBL
    width_256 = get_width(256, GW)
    depth_9 = get_depth(9, GD)
    conv3 = convBlock(network, weights, c3_2.get_output(0), width_256, 3, 2, 1,"model.3")
    # CSP1_3
    c3_4 = C3(network, weights, conv3.get_output(0), width_256, width_256, depth_9, True, 1, 0.5, "model.4")
    # CBL
    width_512 = get_width(512, GW)
    conv5 = convBlock(network, weights, c3_4.get_output(0), width_512, 3, 2, 1, "model.5")
    # CSP1_3
    c3_6 = C3(network, weights, conv5.get_output(0), width_512, width_512, depth_9, True, 1, 0.5, "model.6")
    # CBL
    width_1024 = get_width(1024, GW)
    conv7 = convBlock(network, weights, c3_6.get_output(0), width_1024, 3, 2, 1, "model.7")
    # SPP
    spp8 = SPP(network, weights, conv7.get_output(0), width_1024, width_1024, 5, 9, 13, "model.8")

    # --------head-------#
    c3_9 = C3(network, weights, spp8.get_output(0), width_1024, width_1024, depth_3, False, 1, 0.5, "model.9")
    conv10 = convBlock(network, weights, c3_9.get_output(0), width_512, 1,1,1, "model.10")

    #first upsample,32x->upsample->16x
    upsample11 = network.add_resize(conv10.get_output(0))
    assert upsample11, "Add upsample11 failed"
    upsample11.resize_mode = trt.ResizeMode.NEAREST
    upsample11.shape = c3_6.get_output(0).shape

    #Concat
    input_tensors12 = [upsample11.get_output(0), c3_6.get_output(0)]
    cat12 = network.add_concatenation(input_tensors12)
    c3_13 = C3(network, weights, cat12.get_output(0), width_1024, width_512, depth_3, False, 1, 0.5, "model.13")
    conv14 = convBlock(network, weights, c3_13.get_output(0), width_256, 1, 1, 1, "model.14")

    #second upsample,16x->upsample->8x
    upsample15 = network.add_resize(conv14.get_output(0))
    assert upsample15, "Add upsample15 failed"
    upsample15.resize_mode = trt.ResizeMode.NEAREST
    upsample15.shape = c3_4.get_output(0).shape

    #Concat
    input_tensors16 = [upsample15.get_output(0), c3_4.get_output(0)]
    cat16 = network.add_concatenation(input_tensors16)
    c3_17 = C3(network, weights, cat16.get_output(0), width_512, width_256, depth_3, False, 1, 0.5, "model.17")
    det0 = network.add_convolution_nd(c3_17.get_output(0), 3 * (CLASS_NUM + 5), trt.DimsHW(1, 1),
                                      weights["model.24.m.0.weight"], weights["model.24.m.0.bias"])

    #The second branch
    conv18 = convBlock(network, weights, c3_17.get_output(0), width_256, 3, 2, 1, "model.18")
    input_tensors19 = [conv18.get_output(0), conv14.get_output(0)]
    cat19 = network.add_concatenation(input_tensors19)
    c3_20 = C3(network, weights, cat19.get_output(0), width_512, width_512, depth_3, False, 1, 0.5, "model.20")
    det1 = network.add_convolution_nd(c3_20.get_output(0), 3 * (CLASS_NUM + 5), trt.DimsHW(1, 1),
                                      weights["model.24.m.1.weight"], weights["model.24.m.1.bias"])

    #The third branch
    conv21 = convBlock(network, weights, c3_20.get_output(0), width_512, 3, 2, 1, "model.21")
    input_tensors22 = [conv21.get_output(0), conv10.get_output(0)]
    cat22 = network.add_concatenation(input_tensors22)
    c3_23 = C3(network, weights, cat22.get_output(0), width_1024, width_1024, depth_3, False, 1, 0.5, "model.23")
    det2 = network.add_convolution_nd(c3_23.get_output(0), 3 * (CLASS_NUM + 5), trt.DimsHW(1, 1),
                                      weights["model.24.m.2.weight"], weights["model.24.m.2.bias"])

    dets = [det0.get_output(0),det1.get_output(0),det2.get_output(0)]
    yolo = addYoloLayer_v2(network, weights, "model.24", dets)
    assert yolo, "Add Yolo failed"

    return network


def build_engine(wts_name, engine_file=''):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config:
        config.max_workspace_size = GiB(3)
        # default = 1 for fixed batch size
        builder.max_batch_size = 1

        if builder.platform_has_fast_fp16 and USE_FP16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif builder.platform_has_fast_int8 and USE_INT8:
            config.set_flag(trt.BuilderFlag.INT8)
            calibrator = trt.IInt8EntropyCalibrator2()
            # TO DO
            assert 0,'Mode INT8 is not emplemented !'
            # raise NotImplementedError

        # Populate the network using weights from the PyTorch model.
        net = yolov5(network, wts_name)
        # Build and return an engine.
        print("===> Creating Tensorrt Engine...")
        engine = builder.build_engine(net, config)
        if engine:
            with open(engine_file, "wb") as f:
                f.write(engine.serialize())
            return engine
        else:
            return False


def get_engine(wts_name, engine_file, load=True):
    """Load engine when engine file exists else build"""
    if load:
        with open(engine_file, "rb") as f, \
                trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
    else:
        engine = build_engine(wts_name, engine_file)
    return engine


def parse_arg(argc: int, argv):
    if argc < 4:
        return False
    if argv[1] == "-s" and (argc==5 or argc ==7):
        wts = argv[2]
        engine = argv[3]
        net = argv[4]
        if net[0] == 's':
            gd = 0.33
            gw = 0.50
        elif net[0] == 'm':
            gd = 0.67
            gw = 0.75
        elif net[0] == 'l':
            gd = 1.0
            gw = 1.0
        elif net[0] == 'x':
            gd = 1.33
            gw = 1.25
        elif net[0] == 'c' and argc == 7:
            gd = float(argv[5])
            gw = float(argv[6])
        else:
            return False
        if len(net) == 2 and net[1] == '6':
            is_p6 = True
    elif argv[1] == '-d' and argc == 4:
        engine = argv[2]
        img_dir = argv[3]
    else:
        return False
    return wts, engine, is_p6, gd, gw, img_dir


def main(wts_name, engine_file):
    PLUGIN_LIBRARY = "./build/libyolo_plugins.so"
    ctypes.CDLL(PLUGIN_LIBRARY)

    flag = build_engine(wts_name, engine_file)
    assert flag, "Build engine failed"
    print("===> Serialized Engine Saved at: ", engine_file)


if __name__ == "__main__":
    wts_file = "yolov5s.wts"
    engine_file = "yolov5s.engine"
    main(wts_file, engine_file)