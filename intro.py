
# coding: utf-8

# In[37]:


import numpy as np

import time
import datetime
import torch
import torch.nn as nn
import torchvision.datasets as dsets 
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader


# In[12]:


train_dataset = dsets.MNIST(root ='./data',  
                            train = True,  
                            transform = transforms.ToTensor(), 
                            download = True)
  
test_dataset = dsets.MNIST(root ='./data',  
                           train = False,  
                           transform = transforms.ToTensor(),
                           download = True)


# In[24]:


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True,
                          num_workers=n_cpu)


# In[25]:


test_loader = DataLoader(dataset = test_dataset,  
                         batch_size = batch_size,  
                         shuffle = False,
                         num_workers=1) 


# ### Declare variables for hyper parameters

# In[23]:


input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
n_cpu = 8 # number of logical cpu


# ### Check if your environment allows to use GPU

# In[33]:


cuda = True if torch.cuda.is_available() else False


# ### Create your own model

# In[31]:


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        predict = self.linear(x)
        return predict


# ### Initialize the model and setup loss function and optimizer

# In[36]:


model = LogisticRegression(input_size, num_classes)
if cuda:
    model.cuda()
    print("GPU computation is enabled!")
else:
    print("No cuda device is available!")

# Loss Function
criterion = nn.CrossEntropyLoss()

# Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


# ### Training

# In[41]:


prev_time = time.time()
for epoch in range(num_epochs): 
    for i, (images, labels) in enumerate(train_loader):
        
        # transform 
        images = Variable(images.view(-1, 28 * 28)) 
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(images)
        
        # compute batch loss
        loss = criterion(outputs, labels)
        
        # backpropagation
        loss.backward()
        
        # optimize
        optimizer.step() 

        batches_done = epoch * len(train_loader) + i
        batches_left = num_epochs * len(train_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        
        if (i + 1) % 100 == 0: 
            print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f, ETA: %s'
                  % (epoch + 1, num_epochs, i + 1, 
                     len(train_dataset) // batch_size, loss.item(), time_left))


# In[30]:


correct = 0
total = 0
for images, labels in test_loader: 
    images = Variable(images.view(-1, 28 * 28)) 
    outputs = model(images) 
    _, predicted = torch.max(outputs.data, 1) 
    total += labels.size(0) 
    correct += (predicted == labels).sum() 
  
print('Accuracy of the model on the 10000 test images: % d %%' % ( 
            100 * correct / total)) 


# Ref: https://www.geeksforgeeks.org/identifying-handwritten-digits-using-logistic-regression-pytorch/
