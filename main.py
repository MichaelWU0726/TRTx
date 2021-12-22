from lib import main_helmetfire_fixed


if __name__ == "__main__":
    engine_file = 'onnx_engine/helmetfire_yolov5_fixed_640.engine'
    onnx_file = 'onnx_engine/helmetfire_yolov5_fixed_sim.onnx'
    main_helmetfire_fixed(engine_file=engine_file, onnx_file=onnx_file)