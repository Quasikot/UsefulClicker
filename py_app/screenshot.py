# -*- coding: utf-8 -*-


import sys
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage


def take_screenshot():
  """Takes a screenshot of the entire screen and saves it to a file."""

  #app = QtWidgets.QApplication(sys.argv)
  app = QtWidgets.QApplication.instance()
  screen = app.primaryScreen()
  screenshot = QImage(screen.grabWindow(app.desktop().winId()))
  screenshot.save("screenshot.png", "png")
  mat = cv2.imread("screenshot.png")
 # mat = mat.reshape(screenshot.height(), screenshot.width(), screenshot.depth() // 8)
  return mat


