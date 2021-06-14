
# plot feature map of first conv layer for given image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
import os
import numpy as np

path = 'FMaps2/'
layer = 16
# load the model
model = VGG16()
cnn_name = model.name
layer_name = model.layers[layer].name

# redefine model to output right after the first hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[layer].output)
model.summary()
# load the image with the required shape
img = load_img('C_6.jpg', target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot all 64 maps in an 8x8 squares

fmap_matrix = feature_maps[0]
fmaps = np.array()
for k in range(0, 512):
    fmap = []
    for i in range(0, len(fmap_matrix)):
        fmap_row = []
        for j in range(0, len(fmap_matrix)):
            fmm = fmap_matrix[k]
            fmap_row.append(fmap_matrix[k][i][j])
        fmap.append(fmap_row)
    np_fmap = np.array(fmap)
    fmaps.append(np_fmap)




fmap_count = model.output_shape[3]
