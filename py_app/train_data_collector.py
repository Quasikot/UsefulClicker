# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from screenshot import take_screenshot
from preprocess import canny_edge_detection
import torch
import torchvision.models as models
import torchvision.transforms as transforms

import sys
from PyQt5.QtGui import QImage, QPainter
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt  
from PyQt5.QtGui import QFontDatabase, QFont, QColor
  

def prepare_char_data():
  # Read the image.
  image = cv2.imread("Screenshot_english_alphabet.png")#take_screenshot()
  image = cv2.resize(image, (image.shape[1]*4, image.shape[0]*4))

  # Detect edges on the image.
  edges = canny_edge_detection(image, 10, 100)
  
  # Find the contours in the image.
  contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Draw the filtered contours on the image.
  index = 0
  for contour in contours:
      # Generate a random color for each contour.
    random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    bounding_rect = cv2.boundingRect(contour)
    x, y, width, height = bounding_rect
    cropped_image = image[y:y + height, x:x + width]
    #cv2.imwrite(f"char_data\{index}.png", cropped_image)
    cv2.drawContours(image, [contour], 0, random_color, 1)
    index = index + 1
    
  cv2.imwrite("contours.png", image)
  
def prepare_char_data_qt():
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_+={}[]:;<>,.?/'
    app = QtWidgets.QApplication(sys.argv)
    db = QFontDatabase()
    fonts = [font for font in db.families()]
    print(fonts)
    i = 0
    character_index = 0
    image = QImage(32, 32, QImage.Format_RGB888)
    painter = QPainter(image)
    for character in chars:
        #print(character)
        for font in fonts:
          f = QFont(font)
          f.setPixelSize(30)
          painter.setFont(f)
          painter.setPen(Qt.black)
          painter.fillRect(image.rect(), Qt.white)
          painter.drawText(image.rect(), Qt.AlignCenter, character)
          image.save(f"char_data\{character_index}_{i}.png")
          i = i + 1
        character_index = character_index + 1
#prepare_char_data()
prepare_char_data_qt()
#image = cv2.imread("char_data\\7226.png")

