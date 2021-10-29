import struct
import sys
from typing import List, Any

import numpy as np
import tensorrt as trt


class Yolo(object):
    INPUT_H, INPUT_W = 640, 640  # // yolov5's input height and width must be divisible by 32
    DTYPE = trt.float32
    CHECK_COUNT = 3

    CLASS_NUM = 80
    MAX_OUTPUT_BBOX_COUNT = 1000
    GW: float = 0.50
    GD: float=0.33


class YoloKernel(object):
    width = 256
    height = 256
    anchors = []


def loadWeights(wts_file:str):
    print("Loading weights: ", wts_file)
    weights = {}  # str:trt.Weights
    try:
        with open(wts_file) as f:
            count = int(f.readline())
            assert count>0, "Invalid weight file."
            for i in range(count):
                f_data = f.readline().split()
                name, size = f_data[:2]
                wts = []
                for i in range(int(size)):
                    w = f_data[i+2]
                    w_ = struct.unpack(">f", bytes.fromhex(w))[0]
                    wts.append(w_)
                val = np.array(wts, dtype=np.float32)
                wt = trt.Weights(val)
                # wt.dtype = trt.DataType.FLOAT
                # wt.size = size
                # print("wt.dtype: ", wt.dtype, "\twt.size: ", wt.size)
                weights[name] = wt
            print("Finished Load weights: ", wts_file)
    except FileNotFoundError:
        print("Unable to load weight file.")
        sys.exit(1)
    return weights


def addBatchNorm2d(network,weights,inp,lname: str,eps):
    gamma = weights[lname+".weight"].numpy()
    beta = weights[lname+".bias"].numpy()
    mean = weights[lname+".running_mean"].numpy()
    var = weights[lname+".running_var"].numpy()
    count = len(weights[lname + ".running_var"])
    in_var = np.sqrt(var+eps)

    #scale
    scval = gamma/in_var
    scale = trt.Weights(scval)
    shval = beta-mean*gamma/in_var
    shift = trt.Weights(shval)
    pval = np.ones(count, dtype=np.float32)
    power = trt.Weights(pval)

    weights[lname + ".scale"] = scale
    weights[lname + ".shift"] = shift
    weights[lname + ".power"] = power
    scale_1 = network.add_scale(inp, mode=trt.ScaleMode.CHANNEL, shift=shift, scale=scale, power=power)
    assert scale_1, "Add scale layer failed"

    return scale_1

def convBlock(network, weights, inp, outch: int, ksize: int, s: int, g, lname):
    conv1_w = weights[lname + ".conv.weight"].numpy()
    conv1_b = trt.Weights(trt.float32)
    p = ksize//2
    conv1 = network.add_convolution_nd(inp, num_output_maps=outch, kernel_shape=trt.DimsHW(ksize, ksize), kernel=conv1_w, bias=conv1_b)
    assert conv1, "Add convolution_nd layer failed"
    conv1.stride_nd = trt.DimsHW(s, s)
    conv1.padding_nd = trt.DimsHW(p, p)
    conv1.num_groups = g
    bn1 = addBatchNorm2d(network, weights, conv1.get_output(0), lname+".bn", 1e-3)

    # silu = x * sigmoid
    sig = network.add_activation(bn1.get_output(0), trt.ActivationType.SIGMOID)
    assert sig, "Add activation layer failed"
    ew = network.add_elementwise(bn1.get_output(0), sig.get_output(0), trt.ElementWiseOperation.PROD)
    assert ew, "Add PROD layer failed"

    return ew

def addFusedBatchNorm2d(network, weights, inp, outch, ksize, stride, group: int, lname: str):
    """
    卷积和BN层融合版 https://blog.csdn.net/github_28260175/article/details/103515033
    """
    conv_w = weights[lname+'.conv.weight'].numpy()
    # conv_b = weights[lname+'.conv.bias'].numpy()
    # conv_b = trt.Weights(trt.float32)
    conv_b = np.zeros(conv_w.shape[0], np.float32)  # 该卷积层没有bias
    gamma = weights[lname+'.bn.weight'].numpy()  # bn gamma
    beta = weights[lname+'.bn.bias'].numpy()  # bn beta
    mean = weights[lname+'.bn.running_mean'].numpy()  # bn mean
    var = weights[lname+'.bn.running_var'].numpy()  # bn var
    eps = 1e-05
    bn_var = np.sqrt(var + eps)

    fused_conv_w = conv_w * (gamma / bn_var).reshape([conv_w.shape[0], 1, 1, 1])
    fused_conv_b = (conv_b - mean) / bn_var * gamma + beta
    fused_conv = network.add_convolution_nd(inp, num_output_maps=outch,
                                         kernel_shape=(ksize, ksize), kernel=fused_conv_w,
                                         bias=fused_conv_b)
    pad = ksize//2
    fused_conv.padding_nd = (pad, pad)  # 卷积的pad
    fused_conv.stride_nd = (stride, stride)  # 卷积的stride
    fused_conv.num_groups = group

    return fused_conv

def fusedConvBlock(network, weights, inp, outch: int, ksize: int, stride: int, group: int, lname: str):
    """CBL
    Conv + BN + LeakyRELU
    """
    fused_cv = addFusedBatchNorm2d(network, weights, inp, outch, ksize, stride, group, lname+".conv")
    leaky_relu = network.add_activation(fused_cv.get_output(0), trt.ActivationType.Leaky_RELU)

    return leaky_relu

def focus(network, weights, inp, inch, outch, ksize, lname):
    shape = trt.Dims3(inch, Yolo.INPUT_H//2, Yolo.INPUT_W//2)
    stride = trt.Dims3(1,2,2)
    s1 = network.add_slice(inp, trt.Dims3(0,0,0), shape, stride)
    s2 = network.add_slice(inp, trt.Dims3(0,1,0), shape, stride)
    s3 = network.add_slice(inp, trt.Dims3(0,0,1), shape, stride)
    s4 = network.add_slice(inp, trt.Dims3(0,1,1), shape, stride)
    input_tensors = [s1.get_output(0), s2.get_output(0), s3.get_output(0), s4.get_output(0)]
    cat = network.add_concatenation(input_tensors)  # #通道维度上的拼接
    # conv = fusedConvBlock(network, weights, cat.get_output(0), outch, ksize, 1, 1, lname)
    conv = convBlock(network, weights, cat.get_output(0), outch, ksize, 1, 1, lname + ".conv")

    return conv


def bottleneck(network, weights, inp, c1: int, c2: int, shortcut: bool, g: int, e: int, lname: str):
    conv1 = convBlock(network, weights, inp, int(float(c2)*e), 1,1,1, lname+".cv1")
    conv2 =convBlock(network, weights, conv1.get_output(0), c2, 3,1,g, lname+".cv2")
    if shortcut and c1 == c2:
        ew = network.add_elementwise(inp, conv2.get_output(0), op=trt.ElementWiseOperation.SUM)
        return ew
    return conv2

def bottleneckCSP(network, weights, inp, c1,c2,n,shortcut,g,e,lname):
    conv_w = weights[lname+".cv2.weight"]
    conv_b = trt.Weights(trt.float32)
    c_ = int(float(c2)*e)
    conv1 = convBlock(network, weights, inp, c_, 1,1,1,lname+".cv1")
    conv2 = network.add_convolution_nd(inp, c_, trt.DimsHW(1,1), conv_w, conv_b)
    y1 = conv1.get_output(0)
    for i in range(n):
        b = bottleneck(network, weights, y1, c_, c_, shortcut, g, 1.0, lname+".m."+str(i))
        y1 = b.get_output(0)
    conv_w = weights[lname+".cv3.weight"]
    conv3 = network.add_convolution_nd(y1, c_, trt.DimsHW(1,1), conv_w, conv_b)

    input_tensors = [conv3.get_output(0), conv2.get_output(0)]
    cat = network.add_concatenation(input_tensors)

    bn = addBatchNorm2d(network, weights, cat.get_output(0), lname+".bn", 1e-4)
    lr = network.add_activation(bn.get_output(0), trt.ActivationType.Leaky_RELU)
    lr.alpha = 0.1

    conv4 = convBlock(network, weights, lr.get_output(0), c2, 1,1,1, lname+".cv4")
    return conv4

def C3(network, weights, inp, c1, c2, n, shortcut, g, e, lname):
    c_ = int(float(c2)*e)  # e:expand param
    conv1 = convBlock(network, weights, inp, c_, 1, 1,1, lname+".cv1")
    conv2 = convBlock(network, weights, inp, c_, 1, 1,1, lname+".cv2")
    y1 = conv1.get_output(0)
    for i in range(n):
        b = bottleneck(network, weights, y1, c_, c_, shortcut, g, 1.0, lname + ".m." + str(i))
        y1 = b.get_output(0)

    input_tensors = [y1, conv2.get_output(0)]
    cat = network.add_concatenation(input_tensors)

    conv3 = convBlock(network, weights, cat.get_output(0), c2, 1,1,1, lname+".cv3")
    return conv3

def SPP(network, weights, inp, c1, c2, k1, k2,k3, lname):
    c_ = c1//2
    conv1 = convBlock(network, weights, inp, c_, 1,1,1, lname+".cv1")
    pool1 = network.add_pooling_nd(conv1.get_output(0), trt.PoolingType.MAX, trt.DimsHW(k1,k1))
    pool1.padding_nd = trt.DimsHW(k1//2, k1//2)
    pool1.stride_nd = trt.DimsHW(1,1)
    pool2 = network.add_pooling_nd(conv1.get_output(0), trt.PoolingType.MAX, trt.DimsHW(k2, k2))
    pool2.padding_nd = trt.DimsHW(k2 // 2, k2 // 2)
    pool2.stride_nd = trt.DimsHW(1, 1)
    pool3 = network.add_pooling_nd(conv1.get_output(0), trt.PoolingType.MAX, trt.DimsHW(k3, k3))
    pool3.padding_nd = trt.DimsHW(k3 // 2, k3 // 2)
    pool3.stride_nd = trt.DimsHW(1, 1)

    input_tensors = [conv1.get_output(0), pool1.get_output(0), pool2.get_output(0), pool3.get_output(0)]
    cat = network.add_concatenation(input_tensors)

    conv2 = convBlock(network, weights, cat.get_output(0), c2, 1,1,1, lname+".cv2")
    return conv2


def getAnchors(weights, lname):
    wts = weights[lname+".anchor_grid"].numpy()
    anchor_len = Yolo.CHECK_COUNT * 2
    anchors = []
    assert len(wts) % anchor_len == 0, "The num of anchor_grid isn't a multiple of anchor_len"
    anchor_num = len(wts) // anchor_len
    for i in range(anchor_num):
        anchors.append(wts[i * anchor_len:(i+1)*anchor_len])
    return anchors


def addYoloLayer(network, weights, lname, dets):
    plugin_creator = trt.get_plugin_registry().get_plugin_creator('YoloLayer_TRT', "1")
    assert plugin_creator, "Plugin YoloLayer_TRT isn't registried"
    anchors = getAnchors(weights, lname)

    def get_trt_plugin(plugin_name):
        netinfo = np.ascontiguousarray([Yolo.CLASS_NUM, Yolo.INPUT_W, Yolo.INPUT_H, Yolo.MAX_OUTPUT_BBOX_COUNT])
        scale = 8
        kernels = []
        for i in range(len(anchors)):
            kernel = YoloKernel()
            kernel.width = Yolo.INPUT_W / scale
            kernel.height = Yolo.INPUT_H / scale
            kernel.anchors = anchors[i]
            kernels.append(kernel)
            scale *= 2
        kernels = np.ascontiguousarray(kernels)

        field_collect = trt.PluginFieldCollection()

        netinfo_field = trt.PluginField("netinfo", netinfo, trt.PluginFieldType.FLOAT32)
        assert netinfo_field,"create netinfo failed"
        field_collect.append(netinfo_field)

        kernels_field = trt.PluginField("kernels", kernels, trt.PluginFieldType.FLOAT32)
        assert kernels_field,"create kernels failed"
        field_collect.append(kernels_field)

        plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collect)
        assert plugin, "Create {} layer failed".format(plugin_name)
        return plugin

    trt_plugin = get_trt_plugin("yololayer")
    yolo = network.add_plugin_v2(inputs=dets, plugin=trt_plugin)
    return yolo

def addYoloLayer_v2(network, weights, lname, dets):
    # refer to
    # https://github.com/jkjung-avt/tensorrt_demos/blob/ca6ac81b4c7558122df3ff3018479412828a88a4/yolo/plugins.py#L82-L146
    plugin_creator = trt.get_plugin_registry().get_plugin_creator('YoloLayer_TRT', "1")
    assert plugin_creator, "Plugin YoloLayer_TRT isn't registried"
    anchors = getAnchors(weights, lname)

    netinfo = np.array([Yolo.CLASS_NUM, Yolo.INPUT_W, Yolo.INPUT_H, Yolo.MAX_OUTPUT_BBOX_COUNT],dtype=np.int32)
    field_collect = trt.PluginFieldCollection()
    field_collect.append(trt.PluginField("netinfo", netinfo, trt.PluginFieldType.INT32))
    scale = 8
    for i in range(len(anchors)):
        width = Yolo.INPUT_W // scale
        height = Yolo.INPUT_H // scale
        yoloW = np.array(width,dtype=np.int32)
        yoloH = np.array(height,dtype=np.int32)
        anchor = np.ascontiguousarray(anchors[i],dtype=np.float32)
        field_collect.append(trt.PluginField("yoloW", yoloW, trt.PluginFieldType.INT32))
        field_collect.append(trt.PluginField("yoloH", yoloH, trt.PluginFieldType.INT32))
        field_collect.append(trt.PluginField("anchor", anchor, trt.PluginFieldType.FLOAT32))
        scale *= 2

    plugin = plugin_creator.create_plugin(name="yololayer", field_collection=field_collect)
    assert plugin, "Create yololayer plugin failed"

    yolo = network.add_plugin_v2(inputs=dets, plugin=plugin)
    return yolo
