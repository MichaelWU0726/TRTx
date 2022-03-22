import os
import numpy as np
from util import InferErr,GiB,HostDeviceMem
import tensorrt as trt
import pycuda.driver as cuda

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def generate_net(network, weights):
    input_tensor = network.add_input(name="input", dtype=trt.float32, shape=(3, 5, 5))
    print(input_tensor.shape)
     
    # scale_1 = network.add_scale(input=input_tensor, mode=trt.tensorrt.ScaleMode.UNIFORM, shift=np.zeros((1), dtype=np.float32), scale=np.array([4], dtype=np.float32), power=np.ones((1), dtype=np.float32))
    
    # unary_1 = network.add_unary(input=input_tensor, op=trt.tensorrt.UnaryOperation.EXP)
    # print(unary_1.get_output(0).shape)
    
    # elemnet_1 = network.add_elementwise(input1=input_tensor, input2=input_tensor, op=trt.tensorrt.ElementWiseOperation.DIV)
    # print(elemnet_1.get_output(0).shape)
    # reduce_1 = network.add_reduce(input=input_tensor, op=trt.tensorrt.ReduceOperation.MAX, axes=1, keep_dims=True)
    # print(reduce_1.get_output(0).shape)
    # network.mark_output(reduce_1.get_output(0))
    
    # div_1 = network.add_elementwise(input1=input_tensor, input2=reduce_1.get_output(0), op=trt.tensorrt.ElementWiseOperation.DIV)
    # network.mark_output(div_1.get_output(0))
    
    const_1 = network.add_constant(shape=[3], weights=np.array([0,2,1], dtype=np.int32))
    gather_1 = network.add_gather(input=input_tensor, indices=const_1.get_output(0), axis=2)
    print(gather_1.get_output(0).shape)
    network.mark_output(gather_1.get_output(0))
    
    # shuffle_1 = network.add_shuffle(input=gather_1.get_output(0))
    # shuffle_1.reshape_dims=[1, 5, 5]
    # print(shuffle_1.get_output(0).shape)
    
    # concat_1 = network.add_concatenation(inputs=[input_tensor, gather_1.get_output(0)])
    # print(concat_1.get_output(0).shape)
    
    # network.mark_output(elemnet_1.get_output(0))

class GatherFixedTrt:
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)

    def __init__(self, weights, engine_file_path: str, gpu_id=0, using_half=True):
        cuda.init()
        # Create CUDA context
        self.cuda_ctx = cuda.Device(gpu_id).make_context()
        # Prepare the runtine engine
        self.using_half = using_half
        self.engine_path = engine_file_path
        self.engine = self.get_engine(weights)
        self.context = self.engine.create_execution_context()

    def build_engine(self, weights):
        self.builder.max_workspace_size = GiB(1)
        self.builder.max_batch_size = 1
        if self.using_half:
            self.builder.strict_type_constraints = True
            self.builder.fp16_mode = True
            self.builder.int8_mode = False
        # data = self.network.add_input(name='input', dtype=trt.float32, shape=trt.Dims3(3, 2, 3))
        # ind = self.network.add_input(name='indice', dtype=trt.int32, shape=trt.Dims3(3, 2, 3))
        # gather = self.network.add_gather(inputs=data, indices=)
        # lrelu.get_output(0).name = "outputs"
        # network.mark_output(lrelu.get_output(0))
        generate_net(self.network,weights)

        print('Building an engine from weights; this may take a while...')
        engine = self.builder.build_cuda_engine(self.network)
        assert engine, "Build engine failed"
        print("Completed creating Engine")
        with open(self.engine_path, "wb") as f:
            f.write(engine.serialize())
        return engine
    
    def get_engine(self, weights):
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

        if os.path.exists(self.engine_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("===> Reading engine from file {}".format(self.engine_path))
            with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            print("===> build tensorrt engine...")
            return self.build_engine(weights)

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
    
    def __call__(self, inp):
        self.cuda_ctx.push()
        try:
            assert inp.flags['C_CONTIGUOUS'], 'input is not C_CONTIGUOUS'
            if inp.dtype != np.float32:
                inp = inp.astype(np.float32, copy=False)
            inf_list = [inp]  # 模型可能有多个输入所以用list装起来
            outputs = self.do_inference(inf_list)
            return outputs
        except Exception as e:
            raise InferErr(e)
        finally:
            self.cuda_ctx.pop()


def main_gather(engine_file):
    input_array = np.array(range(75), dtype=np.float32).reshape(1,3,5,5) 
    weights = None
    net = GatherFixedTrt(weights, engine_file)
    pred = net(input_array)
    print(pred.shape)