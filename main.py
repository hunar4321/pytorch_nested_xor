#@hunar ahmad abdulrahman

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# True for 1 hidden neural network, False to test 2 hidden layer neural network
one_hidden = True; 
EPOCHS_TO_TRAIN = 1000
input_len = 32


# The following three functions below generates the data for a nexted xor function described by Andrew NG in the following youtube video: 
# https://youtu.be/5dWp1mw_XNk?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&t=381

def gen_xor(x1, x2):
  if(x1==0 and x2==1):
    return 1
  if(x1==1 and x2==0):
    return 1
  else:
    return 0

def loop_xor(xs):
  ys = []  
  i = 0;
  while(i < len(xs)-1):
      ys.append(gen_xor(xs[i], xs[i+1]))
      i = i+2 
  return ys

def nested_xor(xs):
  xs2 = loop_xor(xs)
  xs3 = loop_xor(xs2)
  xs4 = loop_xor(xs3)
  xs5 = loop_xor(xs4)
  out = gen_xor(xs5[0],xs5[1])
  return out

# generate the X and Y data
Xdata=[]
Ydata=[]
for i in range(100): # generate 100 samples
  Xdata.append([])
  for j in range(input_len):
    Xdata[i].append(round(random.random()))
  Ydata.append(nested_xor(Xdata[i]))
        
inputs = list(map(lambda s: Variable(torch.Tensor([s])), Xdata))
targets = list(map(lambda s: Variable(torch.Tensor([s])), Ydata))

class Net(nn.Module):   

    def __init__(self):
        super(Net, self).__init__()
        
        _input = input_len 
        _output = 1 
        
        if(one_hidden == True):
          
          hidden1 = 6

          self.fc1 = nn.Linear(_input, hidden1, True)
          self.fc2 = nn.Linear(hidden1, _output, True)          
          print("One hidden layer, total params:", _input*hidden1 + hidden1*_output)
          
        else:

          hidden1 = 6
          hidden2 = 3

          self.fc1 = nn.Linear(_input, hidden1, True)
          self.fc2 = nn.Linear(hidden1, hidden2, True)
          self.fc3 = nn.Linear(hidden2, _output, True)
          print("Two hidden layers, total_params:", _input*hidden1 + hidden1*hidden2 + hidden2* _output)
          

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        if(one_hidden == False):
          x = torch.sigmoid(self.fc2(x))
          x = self.fc3(x)
        else:
          x = self.fc2(x)
        return x

net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

print("Training loop:")
for idx in range(0, EPOCHS_TO_TRAIN):
    for input, target in zip(inputs, targets):
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    # Does the update
    if idx % 200 == 0:
        print("Epoch {: >8} Loss: {}".format(idx, loss.data.numpy()))

print("")
print("Final results:")
for input, target in zip(inputs, targets):
    output = net(input)
    print("Target:[{}] Predicted:[{}] Error:[{}]".format(
        int(target.data.numpy()[0]),
        round(float(output.data.numpy()[0]), 4),
        round(float(abs(target.data.numpy()[0] - output.data.numpy()[0])), 4)
))
