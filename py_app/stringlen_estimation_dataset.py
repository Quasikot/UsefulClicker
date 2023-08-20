# -*- coding: utf-8 -*-
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
from PyQt5.QtCore import Qt, QRect  
from PyQt5.QtGui import QFontDatabase, QFont, QColor 
from PyQt5.QtGui import QImage, QPainter, QTransform
from PyQt5.QtGui import QFontMetrics
app = QtWidgets.QApplication(sys.argv)

def affine_transform(image, angle, translate, scale_factor):



    image_copy = image.copy()
    image.fill(Qt.white)
    painter = QPainter(image)
    painter.drawImage(translate[0], translate[1], image_copy)


    return image

def split_rectangle(rectangle, n):
  # Splits a rectangle on n smaller rectangles of equal width.
  if n==0:
      return []
  width = int(rectangle.width() / n)
  height = rectangle.height()
  rectangles = []
  for i in range(n):
    x = i * width
    y = 0
    width = rectangle.width() - i * width
    height = rectangle.height()
    rectangles.append(QRect(x, y, width, height))
  return rectangles


def renderString(text, font):
       avg_word_len = 5
       image = QImage(32*avg_word_len, 32, QImage.Format_RGB888)
       painter = QPainter(image)
       f = QFont(font)
       f.setPixelSize(30)
       painter.setFont(f)
       gray_value_bg = random.randint(170, 255)
       gray_value_fg = int(gray_value_bg / 2) + random.randint(-50, 50)
       painter.setPen(QColor(gray_value_fg, gray_value_fg, gray_value_fg))
       painter.fillRect(image.rect(), QColor(gray_value_bg, gray_value_bg, gray_value_bg))
       painter.drawText(image.rect(), Qt.AlignCenter, text)
       
       bounding_rect = painter.boundingRect(image.rect(), Qt.AlignCenter, text)
       cropped_image = image.copy(bounding_rect)
       cropped_image = cropped_image.scaled(image.width(), image.height())
       
       ##----------------------------- this code only for test
       
      # metrics = QFontMetrics(f)

       #w = metrics.width(text)
    
       # x = 0
       # y = 0
       # n = len(text)
       # spacing = w / n
       # for i in range(n):
       #   painter.drawLine(x, 0, int(x + w), image.height())
       #   x += spacing
       #  #-----------------------------
       del painter
       return cropped_image

def remove_items(list_, indices):
  # Create a new list without the items at the specified indices
  new_list = []
  for i, item in enumerate(list_):
      if i not in indices:
          new_list.append(item)
  return new_list



db = QFontDatabase()
indexes_to_remove = [32,60,71,137,165,168,179,254,255,257,258,259]
fonts = [font for font in db.families()]
fonts = remove_items(fonts, indexes_to_remove)
fonts_for_train = fonts
fonts_for_test = fonts[5:10]
  
# train data
word_dictionary = {}
i = 0
filename = "alice_in_wonderland.txt"



# data with transformations   
with open(filename, "r", encoding='utf-8') as f:
 for line in f:
   for word in line.split():
    for font in fonts_for_train:
        word = word.lower()
        if word not in word_dictionary:
            word_dictionary[word] = 0
            image = renderString(word, font)  
            label = len(word)
            image.save(f"string_estimation_data\{label}_{i}.png")
            translate = (random.randint(-int(image.width()/10), int(image.width()/10)), random.randint(-int(image.height()/10), int(image.height()/10)))
            scale = 0.7 + random.random() * 0.3
            new_image = affine_transform(image, 0, translate, scale)
            new_image.save(f"string_estimation_data\{label}_{i}.png")
        word_dictionary[word] += 1   
        i = i + 1
 f.close()
#image = cv2.imread("char_data\\7226.png")

