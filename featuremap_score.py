
# plot feature map of first conv layer for given image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
#from matplotlib import pyplot
from numpy import expand_dims
import os
import numpy as np
import csv

path = '/home/mehdi/PhD/Data Sets/CLEF2015/CompoundFigureDetectionTraining/NOCOMP/'
#path = 'image/'
layer = 16
# load the model
model = VGG16()
cnn_name = model.name
layer_name = model.layers[layer].name

# redefine model to output right after the first hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[layer].output)
model.summary()

from os import listdir
from os.path import isfile, join

image_files = [f for f in listdir(path) if isfile(join(path, f))]

class_fmap_means = {}
for f in range(0, len(image_files)):
    # load the image with the required shape
    img = load_img(path + image_files[f], target_size=(224, 224))
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
    fmaps = np.array([])
    fmap_means = {}

    for k in range(0, 512):
        fmap = []
        for i in range(0, len(fmap_matrix)):
            fmap_row = []
            for j in range(0, len(fmap_matrix)):
                fmap_row.append(fmap_matrix[i][j][k])
            fmap.append(fmap_row)
        np_fmap = np.array(fmap)
        fmap_mean = np.mean(np_fmap)

        fmap_means.update({k : fmap_mean})
    class_fmap_means.update({image_files[f] : fmap_means})

total_fmap_means = {}
for i in range(0, 512):
    filter_mean_sum = 0
    for j in range(len(image_files)):
        #iterate over fmaps
        filter_means = []
        filter_mean_in_file = class_fmap_means[image_files[j]][i]
        filter_mean_sum += filter_mean_in_file
    overall_filter_mean = filter_mean_sum/len(image_files)
    total_fmap_means.update({i:overall_filter_mean})


total_fmap_means_sorted = {k: v for k, v in sorted(total_fmap_means.items(), key=lambda item: item[1], reverse=True)}

with open('total_fmap_means_nocomp_a.csv', 'w', newline='') as csvfile:
    resultcsv = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    resultcsv.writerow(['Filter Number', 'Mean Activation in Class'])
    for key in total_fmap_means_sorted:
        resultcsv.writerow([key, total_fmap_means_sorted[key]])

major_filters_count = 0
for i in range(0, len(total_fmap_means_sorted)):
    if (total_fmap_means_sorted[i] > 0):
        major_filters_count += 1
print('Number of filters with a total mean above 0:' + str(major_filters_count))
major_filters_count = 0
for i in range(0, len(total_fmap_means_sorted)):
    if (total_fmap_means_sorted[i] > 1):
        major_filters_count += 1
print('Number of filters with a total mean above 1:' + str(major_filters_count))

major_filters_count = 0
for i in range(0, len(total_fmap_means_sorted)):
    if (total_fmap_means_sorted[i] > 10):
        major_filters_count += 1

print('Number of filters with a total mean above 10:' + str(major_filters_count))
