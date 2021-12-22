import os
import tensorrt as trt
import pycuda.driver as cuda
from util import HostDeviceMem, GiB
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


class FixedBase:
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    # make order consistency via onnx.
    # input_names = []
    # output_names = []

    def __init__(self, engine_file_path, onnx_path, *, gpu_id=0, using_half=True):
        cuda.init()
        # Create CUDA context
        self.cuda_ctx = cuda.Device(gpu_id).make_context()
        # Prepare the runtine engine
        self.using_half = using_half
        self.engine_path = engine_file_path
        self.engine = self.get_engine(onnx_path)
        self.context = self.engine.create_execution_context()
        # self.binding_names = self.input_names + self.output_names

    def build_engine(self, onnx_file_path):
        """Takes an ONNX file and creates a TensorRT engine to run inference"""
        with trt.OnnxParser(self.network, TRT_LOGGER) as parser:
            self.builder.max_workspace_size = GiB(1)
            self.builder.max_batch_size = 1
            if self.using_half:
                self.builder.strict_type_constraints = True
                self.builder.fp16_mode = True
                self.builder.int8_mode = False
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print('Parser Error: ',parser.get_error(error))
                    return None
            print('Completed parsing of ONNX file')
            last_layer = self.network.get_layer(self.network.num_layers - 1)
            # Check if last layer recognizes it's output
            if not last_layer.get_output(0):
                print('last layer is not output, marking the output')
                # If not, then mark the output using TensorRT API
                self.network.mark_output(last_layer.get_output(0))
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            # self.network.get_input(0).shape = [1, 3, 640, 640]
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = self.builder.build_cuda_engine(self.network)
            print("Completed creating Engine")
            with open(self.engine_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    def get_engine(self, onnx_file_path):
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

        if os.path.exists(self.engine_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("===> Reading engine from file {}".format(self.engine_path))
            with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            print("===> build tensorrt engine...")
            return self.build_engine(onnx_file_path)

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            binding_idx = self.engine[binding]
            if binding_idx == -1:
                print("Error Binding Names!")
                break
            dims = self.engine.get_binding_shape(binding)
            size = trt.volume(dims) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference(self, inf_list):
        inputs, outputs, bindings, stream = self.allocate_buffers()
        # Transfer input data to the GPU.
        # transfer input data to device
        inputs[0].host = inf_list[0]
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host.copy() for out in outputs]

    def __del__(self):
        self.cuda_ctx.pop()
        del self.cuda_ctx
