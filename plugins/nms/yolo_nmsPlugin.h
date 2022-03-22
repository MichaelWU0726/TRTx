#ifndef TRT_YOLO_NMS_PLUGIN_H
#define TRT_YOLO_NMS_PLUGIN_H
#include "kernel.h"
#include "yolo_nmsUtils.h"
#include "plugin.h"
#include <string>
#include <vector>

using namespace nvinfer1::plugin;
namespace nvinfer1
{
namespace plugin
{

class YoloDetectionOutput : public IPluginV2Ext
{
public:
    YoloDetectionOutput(YoloDetectionOutputParameters param);

    YoloDetectionOutput(YoloDetectionOutputParameters param, int C1, int C2, int numPriors);

    YoloDetectionOutput(const void* data, size_t length);

    ~YoloDetectionOutput() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    IPluginV2Ext* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

    void detachFromContext() override;

private:
    YoloDetectionOutputParameters param;
    int C1, C2, numPriors;
    std::string mPluginNamespace;
};

class YoloNMSPluginCreator : public BaseCreator
{
public:
    YoloNMSPluginCreator();

    ~YoloNMSPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;

    // Parameters for DetectionOutput
    YoloDetectionOutputParameters params;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_YOLO_NMS_PLUGIN_H