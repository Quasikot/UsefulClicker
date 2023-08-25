# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:01:57 2023

@author: admin
"""
import torch
from torch.nn import ReLU, LogSoftmax
from torch import flatten
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import os
from torchvision.io import read_image, ImageReadMode
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from Levenshtein import distance
from preprocess import char_segmentation
from gui import get_words_window

# Create a custom dataset class
class CharDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
       # self.transform = transforms.Compose([transforms.ToTensor()])
        self.images = []
        self.labels = []

        for filename in os.listdir(root_dir):
            image_path = os.path.join(root_dir, filename)
            label = int(filename.split('_')[0])
            
            self.images.append(image_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index] # Image.open(image_path) #
        image =  read_image(image_path, ImageReadMode.GRAY)
        charIdx = image_path.split('_')[1].split(".")[0]
        image = image.float()
        image = image.to('cuda')
       

        label = torch.tensor(self.labels[index])
        label = label.to('cuda')

        return charIdx, image_path, image, label

num_classes = 154
# Define the CNN model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(5,5), stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.relu1 = ReLU()
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(2,2), stride=1, padding=0)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.relu2 = ReLU()
        self.conv3 = torch.nn.Conv2d(64, 120, kernel_size=(2,2), stride=1, padding=0)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.relu33 = ReLU()
        self.conv4 = torch.nn.Conv2d(120, 240, kernel_size=(2,2), stride=1, padding=0)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.relu44 = ReLU()
        
        self.fc1 = torch.nn.Linear(240, 1200)
        self.flat = torch.nn.Flatten()
        self.relu3 = ReLU()
        self.fc2 = torch.nn.Linear(in_features=1200, out_features=600)
        self.relu4 = ReLU()
        self.fc3 = torch.nn.Linear(in_features=600, out_features=num_classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
  		# POOL layers
        x= x.cuda()
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
       
  		# pass the output from the previous layer through the second
  		# set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
 
        x = self.conv3(x)
        x = self.relu33(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.relu44(x)
        x = self.pool4(x)
        
  		# flatten the output from the previous layer and pass it
  		# through our only set of FC => RELU layers
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
  		# pass the output to our softmax classifier to get our output
  		# predictions
        output = self.logSoftmax(x)
        return x

def CNNRecognitionTest1():
    rects=char_segmentation()
    
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя!@#$%^&*()-_+={}[]:;<>,.?/'   
    # # Test the model
    model = CNN()
    # Move the module to GPU
    
    model = model.to('cuda')
    model = torch.load("models\\english_chars_cnn.model")
    test_dataset = CharDataset(root_dir='preprocess//chars')
    train_loader = data_utils.DataLoader(test_dataset, batch_size=32, shuffle=False)
    # test_data = dataset.test_data.transform(transform)
    # test_labels = dataset.test_labels
    
    correct = 0
    total = 0
    
    words = {}
    
    print("OCR in progress...")
    
    for i, (charIdxs, image_paths, images, labels) in enumerate(train_loader):
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        #print(f"labels={len(labels)}")
        #print(outputs.data[0])
        #im = cv2.imread(image_paths[0])
        #plt.imshow(im)
        #plt.show()
        #print(chars[predicted[0]])
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i, label in enumerate(labels):
            if label.item() not in words:
                words[label.item()] = []
                #print(label.item())
            words[label.item()].append((int(charIdxs[i]),chars[predicted[i]]))
            
    def sort_pairs(pairs):
      """Sorts a list of pairs by the first element of the pair."""
      pairs.sort(key=lambda pair: pair[0])
      return pairs
    
    words2 = {}
    for key in words:
        pairs = sort_pairs(words[key])
        string=""
        for c in pairs:
            string+=c[1]
        print(f"{key}:{string}")
        words2[key] = string
    
    return words2, rects
    
    
    # # fix words by Levenshtein distance
    # with open('data\\en_US-large.txt', 'r', encoding='utf-8') as f:
    #     # Get a list of all the lines in the file
    #     lines = f.readlines()
        
    # for key in words:
    #   min_distance=202002020;
    #   word_min = ""
    #   for voc_word in lines:
    #       voc_word = voc_word.strip("\n")
    #       d = distance(words[key], voc_word)
    #       if d < min_distance:
    #           word_min = voc_word
    #           min_distance = d
    #   words[key] = word_min
              
        
        #print('Accuracy: {}%'.format(100 * correct / total))
words, rects = CNNRecognitionTest1()
get_words_window(words, rects)