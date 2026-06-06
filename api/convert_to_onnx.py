import tf2onnx

tflite_path = "model.tflite"
onnx_path = "model.onnx"

# Perform conversion
tf2onnx.convert.from_tflite(tflite_path, output_path=onnx_path)
