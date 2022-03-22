#ifndef TRT_YOLO_NMS_UTILS_H
#define TRT_YOLO_NMS_UTILS_H

#include "plugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;

size_t detectionInferenceWorkspaceSize(bool shareLocation, int N, int C1, int C2, int numClasses, int numPredsPerClass,
    int topK, DataType DT_BBOX, DataType DT_SCORE);
#endif