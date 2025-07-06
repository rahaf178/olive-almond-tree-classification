# olive-almond-tree-classification

# To import the necessary libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array # img_to_array : To convert the image into a number matrix so that the model can understand it.
from tensorflow.keras.layers import DepthwiseConv2D
import numpy as np
from google.colab import files

# Fixes an issue when loading a model with a DepthwiseConv2D layer containing groups, by ignoring them.
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None) 
        super().__init__(*args, **kwargs)

print("Upload the file")
upload = files.upload()
ModelPath = list(upload.keys())[0] # Store the name of the first file uploaded from the dictionary.

model = load_model(ModelPath, custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D}) #To load the model from the file and store it inside the model variable for use in image prediction.
print("The model has been loaded")

print("Upload image")
UploadImage = files.upload() # To load the image
ImagePath = list(UploadImage.keys())[0]

Image = load_img(ImagePath, target_size=(224, 224)) 
img_array = img_to_array(Image) # Convert the image into a form that the model understands.
img_array = np.expand_dims(img_array, axis=0) / 255.0

pred = model.predict(img_array)
class_index = np.argmax(pred, axis=1)[0] # axis=1 --> searches every row (a row represents an image).

class_names = ['olive tree', 'almond tree']

probability = pred[0][class_index] # The probability that the image belongs to the class identified by the model.
print(f"Image : {image_path}")
print(f"Class : {class_names[class_index]} (probability : {probability*100:.2f}%)") # probability*100 : Converts a number from a value between 0 and 1 to a percentage between 0 and 100 /.2f : It means printing the number as a decimal with two digits after the decimal point.
