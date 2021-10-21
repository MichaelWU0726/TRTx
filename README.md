# TRTx

TRTx aims to implement popular deep learning networks with tensorrt network definition Python APIs. As we know, tensorrt has builtin parsers, including caffeparser, uffparser, onnxparser, etc. But when we use these parsers, we often run into some "unsupported operations or layers" problems, especially some state-of-the-art models are using new type of layers.

So why don't we just skip all parsers? Fortunately, Wang-xinyu use TensorRT C++ API to define network, build engine and infer in [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx). But some people are not familiar with C++, Python is a kind of beginner-friendly language and the differences between C++ and Python don't affect the building of engine before inferring.

So why not define network and build engine by TensorRT Python API. I wrote this project to get familiar with tensorrt C++/Python API and also to share and learn from the community.
