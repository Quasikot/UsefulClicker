# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from screenshot import take_screenshot

import sys


def plot_opencv_image(image):

  # Convert the OpenCV image to a NumPy array.
  image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Plot the NumPy array using Matplotlib.
  plt.imshow(image_array)
  plt.show()

def canny_edge_detection(image, low_threshold, high_threshold):


  # Convert the image to grayscale.
  grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply Gaussian blur to the grayscale image to reduce noise.
  blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

  # Apply Canny edge detection to the blurred image.
  edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

  return edges


def convolution_with_long_horizontal_kernel_opencv(image, kernel):

  # Create the output image.
  output_image = image.copy();

  # Apply convolution using OpenCV.
  cv2.filter2D(src=image, ddepth=-1, kernel=kernel, dst=output_image)

  return output_image


def filter_contours_by_max_width(contours, max_width, min_width):

  filtered_contours = []
  for contour in contours:
    # Get the bounding rectangle of the contour.
    bounding_rect = cv2.boundingRect(contour)

    # Check if the width of the bounding rectangle is less than the maximum width.
    if bounding_rect[2] <= max_width and bounding_rect[2] > min_width:
      filtered_contours.append(contour)

  return filtered_contours

def detect_buttons():
  # Read the image.
  image = take_screenshot()

  
  
  # Create the kernel.
  kernel = np.array([[0,   0,    0,  0,   0,    0],
                     [1/3, 2, 5, 5, 2, 1/3],
                     [0,   0,    0,  0,   0,    0]])
  #kernel = kernel.T
                     
  # Detect edges on the image.
  edges = canny_edge_detection(image, 10, 100)
 
  # Creating the kernel(2d convolution matrix)
  #kernel = np.ones((5, 5), np.float32)/30
 # Apply convolution using OpenCV.
  conv_image = convolution_with_long_horizontal_kernel_opencv(edges, kernel)
  
  # Find the contours in the image.
  contours, hierarchy = cv2.findContours(conv_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Filter the contours by maximum width.
  filtered_contours = filter_contours_by_max_width(contours, 150, 10)

  rects = []
  # Draw the filtered contours on the image.
  for contour in filtered_contours:
      # Generate a random color for each contour.
    random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #cv2.drawContours(image, [contour], 0, random_color, 2)
    bounding_rect = cv2.boundingRect(contour)
    rects.append(bounding_rect)
    cv2.rectangle(image, (bounding_rect[0], bounding_rect[1]),
                   (bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3]),
                   random_color, 2)



  # Display the original image and the gradient image.
  plot_opencv_image(image)
  
  cv2.imwrite("filtering.png", image)
  cv2.imwrite("conv_image.png", conv_image)
  
  return rects
 
 # cv2.waitKey(0)
