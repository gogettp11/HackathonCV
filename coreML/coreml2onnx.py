import onnxmltools
import coremltools

# convert archeoHackaton_yolo2_ortho_v5
model_coreml = coremltools.utils.load_spec("archeoHackaton_yolo2_ortho_v5.mlmodel")
model_onnx = onnxmltools.convert.convert_coreml(model_coreml, "Image_Reco")
# Save as text
onnxmltools.utils.save_text(model_onnx, "archeoHackaton_yolo2_ortho_v5.json")
# Save as protobuf
onnxmltools.utils.save_model(model_onnx, "archeoHackaton_yolo2_ortho_v5.onnx")


# # convert archeoHackaton_yolo2_slope_v9
# model_coreml = coremltools.utils.load_spec("archeoHackaton_yolo2_slope_v9.mlmodel")
# model_onnx = onnxmltools.convert.convert_coreml(model_coreml, "Image_Reco")
# # Save as text
# onnxmltools.utils.save_text(model_onnx, "archeoHackaton_yolo2_slope_v9.json")
# # Save as protobuf
# onnxmltools.utils.save_model(model_onnx, "archeoHackaton_yolo2_slope_v9.onnx")
