import time
# import cv2
import numpy as np
from .fixed_base import *
from .add_nms import add_nms
from util import InferErr, check_img_size, LoadImages


class HelmetFireFixedTrt(FixedBase):
    # input_names = ["data"]
    # output_names = ["fc1"]

    def __init__(self, engine_file_path: str, onnx_file: str, gpu_id=0, using_half=True):
        super().__init__(engine_file_path, onnx_file, gpu_id=gpu_id, using_half=using_half)

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
                        print('Parser Error: ', parser.get_error(error))
                    return None
            print('Completed parsing of ONNX file')
            # last_layer = self.network.get_layer(self.network.num_layers - 1)
            last_out = self.network.get_output(0)
            print("unmarking last layer to add nms layer...")
            self.network.unmark_output(last_out)

            nms_layer = add_nms(last_out, self.network)
            nms_layer.get_output(0).name = "num_detections"
            nms_layer.get_output(1).name = "nmsed_boxes"
            nms_layer.get_output(2).name = "nmsed_scores"
            nms_layer.get_output(3).name = "nmsed_classes"
            for i in range(4):
                self.network.mark_output(nms_layer.get_output(i))

            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = self.builder.build_cuda_engine(self.network)
            print("Completed creating Engine")
            with open(self.engine_path, "wb") as f:
                f.write(engine.serialize())
            return engine

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

    @classmethod
    def postprocess(cls, outputs):
        out = outputs[0]
        # feature = feature[:512*batch]  # size of output in h&d is maximum, so it need slice real size.
        # In this case, input shape is fixed so output is full
        # feature = out.reshape((1, 512))
        return out


def main_helmetfire_fixed(source='images/1.mp4',
                      onnx_file='onnx_engine/extract_feature_f32.onnx',
                      engine_file='onnx_engine/extract_fixed_f32.engine', repeat=1):
    strides = [8, 16, 32]
    net = HelmetFireFixedTrt(engine_file, onnx_file, using_half=False)
    stride = int(max(strides))  # model stride
    imgsz = check_img_size(640, s=stride)  # check img_size
    for i in range(10):
        net(np.zeros([1, 3, imgsz, imgsz]))
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    for path, img, im0s, vid_cap in dataset:  # im0s是cv2读取的
        size = img.shape
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        img = np.float32(img) / 255.0  # 0 - 255 to 0.0 - 1.0
        img_batch = np.expand_dims(img, axis=0)  # 增加维数ndim
        img_batch = np.ascontiguousarray(img_batch)

        # Inference
        t1 = time.time()
        pred = net(img_batch)
        t2 = time.time()
        print(f'{size[1]}x{size[2]}', end=' ')
        print(f'({t2 - t1:.3f}s)')


if __name__ == "__main__":
    img_path = "images/cropped_3_4.png"
    main_helmetfire_fixed([img_path])
