# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from screenshot import take_screenshot
import sys
import os

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

def process_word(image, n):
   
    image = cv2.resize(image, (image.shape[1]*4, image.shape[0]*4))
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(grayscale_image.shape)
    # Detect edges on the image.
    edges = cv2.Canny(grayscale_image, 10, 100)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        bounding_rect = cv2.boundingRect(contour)
        bounding_rect = (bounding_rect[0]-10, bounding_rect[1]-10,bounding_rect[2]+10,bounding_rect[3]+10)
      
        cropped_img = grayscale_image[bounding_rect[1]:bounding_rect[1] + bounding_rect[3],bounding_rect[0]:bounding_rect[0]+ bounding_rect[2]]
        if cropped_img.shape[0] > 0 and cropped_img.shape[1]>0 :
            cv2.imwrite(f"preprocess\chars\{n}_{i}.png", cropped_img)
            
def detect_words():
  # Read the image.
  #image = take_screenshot()
  #cv2.imwrite("data\\screenshot.png", image)
  image = cv2.imread("C:\\Users\\admin\\Pictures\\Screenshots\\Screenshot 2023-08-23 165728.png")
  
  
  # Create the kernel.
  kernel = np.array([[0,   0,    0,  ],
                     [2, 2,  2 ],
                     [0,   0,    0]])
 # kernel = kernel.T
                     
  # Detect edges on the image.
  edges = canny_edge_detection(image, 10, 100)
 
  # Creating the kernel(2d convolution matrix)
  #kernel = np.ones((5, 5), np.float32)/30
 # Apply convolution using OpenCV.
  #grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #ret, thresh = cv2.threshold(grayscale_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  conv_image = convolution_with_long_horizontal_kernel_opencv(edges, kernel)
  cv2.imwrite("data\\conv_image.png", conv_image)
  
  
  # Find the contours in the image.
  contours, hierarchy = cv2.findContours(conv_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Filter the contours by maximum width.
  filtered_contours = filter_contours_by_max_width(contours, 300, 10)

  rects = []
  image_canvas = image.copy()
  
  # Draw the filtered contours on the image.
  for i, contour in enumerate(filtered_contours):
      # Generate a random color for each contour.
    random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #cv2.drawContours(image, [contour], 0, random_color, 2)
    bounding_rect = cv2.boundingRect(contour)
    bounding_rect = (bounding_rect[0]-3, bounding_rect[1]-3,bounding_rect[2]+3,bounding_rect[3]+3)
    rects.append(bounding_rect)
    
    cropped_img = image[bounding_rect[1]:bounding_rect[1] + bounding_rect[3],bounding_rect[0]:bounding_rect[0]+ bounding_rect[2]]
   # print(cropped_img.shape)
    # cut word
    if cropped_img.shape[0] > 0 and cropped_img.shape[1]>0 :
        process_word(cropped_img, i)
        cv2.imwrite(f"preprocess\{i}.png", cropped_img)
    cv2.rectangle(image_canvas, (bounding_rect[0], bounding_rect[1]),
                   (bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3]),
                    random_color, 2)
    cv2.putText(image_canvas, f"{i}", (bounding_rect[0]-10, bounding_rect[1]-10), 
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 1, cv2.LINE_AA)




  # Display the original image and the gradient image.
  plot_opencv_image(image_canvas)
  
  cv2.imwrite("data\\words.png", image_canvas)
  
  
  return rects

#detect_words()

# segmenting chars using "dissection" method


for filename in os.listdir("preprocess"):
    image_path = os.path.join("preprocess", filename)
    n_word = filename.split(".")[0]
    if filename=="chars":
        continue
         
    # Convert the input image to grayscale
    #image = cv2.imread("preprocess\\104.png") 
    image = cv2.imread(image_path) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
     # Define the kernel size and anchor point for the erosion filter
    kernel_size = (3, 3)
    anchor_point = (-1, -1)
    ret, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # 
    # Creating kernel
    kernel = np.ones((2, 2), np.uint8)
    # Using cv2.erode() method 
    dissect_array = []
    for i in range(0, thresh.shape[1], 1):
        sum1 = 0
        for j in range(0, thresh.shape[0], 1):
            sum1+=(thresh[j, i]==0)
        dissect_array.append(sum1)
    #plt.plot(dissect_array)
    prev = 0
    lines_x = []
    for i in range(0, len(dissect_array), 1):
        if dissect_array[i] != 0 and prev == 0:
            lines_x.append(i-1)
            zeros_len=0
        prev = dissect_array[i]
    
    x0 = 0
    lines_x.append(thresh.shape[1])
    for n_Char, x in enumerate(lines_x):
        if x0!=0:
            cropped_char = image[0:thresh.shape[1],x0:x]
            if cropped_char.shape[1] >= 2:
                bg = cropped_char[0,0]
                cropped_char = cv2.resize(cropped_char, (20,20))
                large_img = cv2.imread("data\\white.bmp")
                large_img[:,:] = bg
                large_img[6:26,6:26] = cropped_char
                #print(cropped_char.shape)
                cv2.imwrite(f"preprocess\\chars\\{n_word}_{n_Char}.png", large_img)
        #cv2.line(thresh, (x, 0), (x, thresh.shape[1]), (0, 255, 0), 1)
        x0 = x
     # Apply the erosion filter to the grayscale image
    #result = cv2.dilate(gray, kernel=kernel_size, anchor=anchor_point, iterations=1)
    #plot_opencv_image(thresh)
    #break
 # cv2.waitKey(0)

  