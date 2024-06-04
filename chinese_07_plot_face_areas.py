# This script is used to plot images based on RGB values

# Import the necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

# Load a dictionary with pickle
with open('chinese_face_areas.pkl', 'rb') as file:
    faces_areas_dict = pickle.load(file)

# Get the keys of the dictionary
keys = list(faces_areas_dict.keys())

# Get the values of the dictionary
images = list(faces_areas_dict.values())

processed_images = []

# Process the images
for image in images:
    min_val = np.min(image)
    max_val = np.max(image)
    # Normalize the image to [0, 1]
    normalized_image = (image - min_val) / (max_val - min_val)
    # Convert all elements in normalized_image to float32
    normalized_image = normalized_image.astype(np.float32)
    processed_images.append(normalized_image)

# Get the first image
# image = images[0]

# Let's inspect the values of the first image
# print("Shape of the first image: ", image.shape)

# What is the range of values for the three channel?

# range_channel_1 = [np.min(image[:,:,0]), np.max(image[:,:,0])]
# range_channel_2 = [np.min(image[:,:,1]), np.max(image[:,:,1])]
# range_channel_3 = [np.min(image[:,:,2]), np.max(image[:,:,2])]

# print("Range of values for channel 1: ", range_channel_1)
# print("Range of values for channel 2: ", range_channel_2)
# print("Range of values for channel 3: ", range_channel_3)

# print(type(image[0,0,0]))

# value = image[0,0,0]

# print("Value: ", value)
# print("Type of value: ", type(value))


# min_val = np.min(image)
# max_val = np.max(image)

# print("Min value: ", min_val)
# print("Max value: ", max_val)


# Normalize the image to [0, 1]
# normalized_image = (image - min_val) / (max_val - min_val)

# val1 = normalized_image[0,0,0]
# val2 = normalized_image[0,0,1]
# val3 = normalized_image[0,0,2]

# print("Normalized value 1: ", val1)
# print("Normalized value 2: ", val2)
# print("Normalized value 3: ", val3)
# print("Type of normalized value 1: ", type(val1))

# val4 = normalized_image[0,0,:]
# print("Normalized value 4: ", val4)
# print("Type of normalized value 4: ", type(val4))

# Type of normalized image
# print("Type of normalized image: ", type(normalized_image))

# Convert all elements in normalized_image to float32
# normalized_image = normalized_image.astype(np.float32)
# val1 = normalized_image[0,0,0]
# print("Normalized value 1: ", val1)
# print("Type of normalized value 1: ", type(val1))

# print("Type of normalized image: ", type(normalized_image))

# Display the image
# plt.imshow(processed_images[0])
# plt.colorbar()  # Optional, to see the value scaling on the side
# plt.show()


# Display all images in a grid with "keys" as titles
# Check how many images we have
num_images = len(processed_images)
# Create a suitable (non fixed) grid, given the number of images
num_rows = int(np.ceil(num_images / 5))
num_cols = 5
# Create the grid
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
# fig, axs = plt.subplots(2, 5, figsize=(20, 10))
axs = axs.ravel()
for i in range(num_images):
    axs[i].imshow(processed_images[i])
    axs[i].set_title(keys[i])
    axs[i].axis('off')
plt.tight_layout()
plt.show()


# plot the first image based on the RGB values encoded in the three channels
# plt.imshow(image)
# plt.show()


