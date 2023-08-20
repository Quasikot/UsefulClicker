# -*- coding: utf-8 -*-
import torch
from torch.nn import ReLU, LogSoftmax
from torch import flatten
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import os
from torchvision.io import read_image, ImageReadMode
from PIL import Image

# Create a custom dataset class
class StringsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
       # self.transform = transforms.Compose([transforms.ToTensor()])
        self.images = []
        self.labels = []

        for filename in os.listdir(root_dir):
            image_path = os.path.join(root_dir, filename)
            label = int(filename.split('_')[0])
            #print(f"label {label}")

            self.images.append(image_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index] # Image.open(image_path) #
        image =  read_image(image_path, ImageReadMode.GRAY)
       # if self.transform is not None:
       #     image = self.transform(image)
       # image = torch.tensor(image)
       # image = (imagetun).unsqueeze(0)
        image = image.float()
        image = image.to('cuda')
        #print(image.shape)
       

        label = torch.tensor(self.labels[index])
        label = label.to('cuda')

        return image, label

# Transform the training data with geometric transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.1, 0.3)),
])

# Load the custom dataset
dataset = StringsDataset(root_dir='./string_estimation_data', transform=transform)
train_loader = data_utils.DataLoader(dataset, batch_size=32, shuffle=True)

num_classes = 47
#train_data = dataset.transform(transform)

# Define the CNN model
class StringLenEstimationModel(torch.nn.Module):
    def __init__(self):
        super(StringLenEstimationModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 160, kernel_size=(2,2), stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), )
        self.relu1 = ReLU()
        self.conv2 = torch.nn.Conv2d(160, 320, kernel_size=(2,2), stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.relu2 = ReLU()
        self.fc1 = torch.nn.Linear(102400, 500)
        self.flat = torch.nn.Flatten()
        self.relu3 = ReLU()
        self.fc2 = torch.nn.Linear(in_features=500, out_features=num_classes)
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
  		# flatten the output from the previous layer and pass it
  		# through our only set of FC => RELU layers
        x = self.flat(x)
        #print(x.shape)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
  		# pass the output to our softmax classifier to get our output
  		# predictions
        output = self.logSoftmax(x)
        return output

# Create the model instance
model = StringLenEstimationModel()
# Move the module to GPU

model = model.to('cuda')

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model():
    # Train the model
    for epoch in range(3):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
    
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if i % 100 == 0:
                print('Epoch: {} Loss: {:.4f}'.format(epoch, loss.item()))
    
    torch.save(model, "stringlen_estimation.model")


def test_model():
    # # Test the model
    test_dataset = StringsDataset(root_dir='./string_estimation_test_data', transform=transform)
    train_loader = data_utils.DataLoader(test_dataset, batch_size=32, shuffle=True)
    # test_data = dataset.test_data.transform(transform)
    # test_labels = dataset.test_labels
    
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Accuracy: {}%'.format(100 * correct / total))


train_model()
test_model()
  