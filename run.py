from model import Model
import preprocess
import cv2

deblur = Model()
deblur.load_model("model.json", "model.h5")

deblur.predict_deblur('blur.png')