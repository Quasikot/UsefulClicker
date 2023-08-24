# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from screenshot import take_screenshot
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PyQt5.QtCore import Qt, QRect
import sys
from PyQt5.QtGui import QImage, QPainter
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt  
from PyQt5.QtGui import QFontDatabase, QFont, QColor
from PyQt5.QtGui import QImage, QPainter, QTransform

app = QtWidgets.QApplication(sys.argv)

def affine_transform(image, angle, translate, scale_factor):


    
    width, height = image.width(), image.height()

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    resized_image = image.scaled(new_width, new_height)
    image.fill(Qt.white)
    painter = QPainter(image)
    x = (width - new_width) // 2
    y = (height - new_height) // 2
    painter.drawImage(x+translate[0], y+translate[1], resized_image)
    painter.end()


    return image

def renderChar(character, font, bg=None):
    
       image = QImage(32, 32, QImage.Format_RGB888)
       painter = QPainter(image)
       f = QFont(font)
       f.setPixelSize(20)
       painter.setFont(f)
       painter.setPen(Qt.black)
       if random.randint(0,10) < 2:
           f.setItalic(True)
       if random.randint(0,10) < 2:    
           f.setBold(True)
           
       gray_value_bg = random.randint(200, 255)
       gray_value_fg = int(gray_value_bg / 3) 
       #image.fill(QColor(gray_value_bg, gray_value_bg, gray_value_bg))
       painter.setPen(QColor(gray_value_fg, gray_value_fg, gray_value_fg))
       
       if bg==None:
           painter.fillRect(image.rect(), QColor(gray_value_bg, gray_value_bg, gray_value_bg))
       else:
           painter.fillRect(image.rect(), bg)
       painter.drawText(image.rect(), Qt.AlignCenter, character)
       bounding_rect = painter.boundingRect(image.rect(), Qt.AlignCenter, character)
       bounding_rect.setLeft(bounding_rect.left()-3)
       bounding_rect.setWidth(bounding_rect.width()+3)
       image2 = image.copy(bounding_rect).scaled(32,32)
       del painter
       return image2

def remove_items(list_, indices):
  # Create a new list without the items at the specified indices
  new_list = []
  for i, item in enumerate(list_):
      if i not in indices:
          new_list.append(item)
  return new_list

def prepare_char_data_qt():
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_+={}[]:;<>,.?/'
    #chars = 'abf'
    
    db = QFontDatabase()
    indexes_to_remove = [32,60,71,137,165,168,179,254,255,257,258,259]
    fonts = [font for font in db.families()]
    fonts = remove_items(fonts, indexes_to_remove)
    fonts_for_train = fonts
    fonts_for_test = fonts[5:10]
  
    i = 0
    character_index = 0
   
    # train data
    for character in chars:
        for font in fonts_for_train:
          image = renderChar(character, font)  
          image.save(f"char_data\{character_index}_{i}.png")
          i = i + 1
        character_index = character_index + 1
   
    # train data with affine transforms
    # character_index = 0
    # fonts_for_train = fonts_for_train + fonts_for_train
    # for character in chars:
    #     for font in fonts_for_train:
    #       image = renderChar(character, font, bg=Qt.white)
    #       translate = (random.randint(0, int(image.width()/10)), random.randint(0, int(image.height()/10)))
    #       scale = 0.7 + random.random() * 0.3
    #       new_image = affine_transform(image, 0, translate, scale)
  
    #       new_image.save(f"char_data\{character_index}_{i}.png")
    #       i = i + 1
    #     character_index = character_index + 1
        
    # test data
    # i = 0
    # character_index = 0
    # for character in chars:
    #     for font in fonts_for_test:
    #         image = renderChar(character, font)
    #         angle = random.randint(-5, 5)
    #         translate = (random.randint(0, int(image.width()/10)), random.randint(0, int(image.height()/10)))
    #         new_image = affine_transform(image, angle, translate, 1)
    #         new_image.save(f"test_data\{character_index}_{i}.png")
    #         i = i + 1
    #     character_index = character_index + 1

#prepare_char_data()
prepare_char_data_qt()
#image = cv2.imread("char_data\\7226.png")

