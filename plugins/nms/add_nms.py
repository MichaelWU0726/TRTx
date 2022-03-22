import tensorrt as trt
import numpy as np

num_class = 4
topK = 2048
keepTopK = 300
conf_thres = 0.25
iou_thres = 0.45

def add_nms(inp, network):
    # slice boxes, obj_score, class_scores
    strides = trt.Dims([1, 1, 1])
    starts = trt.Dims([0, 0, 0])
    bs, num_boxes, _ = inp.shape
    shapes = trt.Dims([bs, num_boxes, 4])
    boxes = network.add_slice(inp, starts, shapes, strides)
    starts[2] = 4
    shapes[2] = 1
    obj_score = network.add_slice(inp, starts, shapes, strides)
    starts[2] = 5
    shapes[2] = num_class
    scores = network.add_slice(inp, starts, shapes, strides)

    indices = network.add_constant(trt.Dims([num_class]), trt.Weights(np.zeros(num_class, np.int32)))
    gather_layer = network.add_gather(obj_score.get_output(0), indices.get_output(0), 2)

    # scores = obj_score * class_scores => [bs, num_boxes, nc]
    updated_scores = network.add_elementwise(gather_layer.get_output(0), scores.get_output(0),
                                             trt.ElementWiseOperation.PROD)

    # reshape box to [bs, num_boxes, 1, 4]
    reshaped_boxes = network.add_shuffle(boxes.get_output(0))
    reshaped_boxes.reshape_dims = trt.Dims([0, 0, 1, 4])

    # add batchedNMSPlugin, inputs:[boxes:(bs, num, 1, 4), scores:(bs, num, 1)]
    # trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    registry = trt.get_plugin_registry()
    assert registry
    creator = registry.get_plugin_creator("BatchedNMS_TRT", "1")
    assert creator
    fc = [trt.PluginField("shareLocation", np.array([1], dtype=np.int), trt.PluginFieldType.INT32),
          trt.PluginField("backgroundLabelId", np.array([-1], dtype=np.int), trt.PluginFieldType.INT32),
          trt.PluginField("numClasses", np.array([num_class], dtype=np.int), trt.PluginFieldType.INT32),
          trt.PluginField("topK", np.array([topK], dtype=np.int), trt.PluginFieldType.INT32),
          trt.PluginField("keepTopK", np.array([keepTopK], dtype=np.int), trt.PluginFieldType.INT32),
          trt.PluginField("scoreThreshold", np.array([conf_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32),
          trt.PluginField("iouThreshold", np.array([iou_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32),
          trt.PluginField("isNormalized", np.array([0], dtype=np.int), trt.PluginFieldType.INT32),
          trt.PluginField("clipBoxes", np.array([0], dtype=np.int), trt.PluginFieldType.INT32)]

    fc = trt.PluginFieldCollection(fc)
    nms_layer = creator.create_plugin("nms_layer", fc)

    layer = network.add_plugin_v2([reshaped_boxes.get_output(0), updated_scores.get_output(0)], nms_layer)

    return layer