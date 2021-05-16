import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import time

transform=transforms.ToTensor()
train_data=datasets.MNIST(root="Data", train=True, download=True, transform=transform)
test_data=datasets.MNIST(root="Data",train=False, download=True, transform=transform)

image, label= train_data[0]

torch.manual_seed(42)
train_load=DataLoader(train_data, batch_size=100, shuffle=True)
test_load=DataLoader(test_data, batch_size=10000, shuffle=False)

np.set_printoptions(formatter=dict(int=lambda x: f"{x:4}"))

for images, labels in train_load:
    break

class MultilayerPerceptron(nn.Module):
    def __init__(self, in_sz=784, out_sz=10, layers=[120,84]):
        super().__init__()
        self.fc1=nn.Linear(in_sz,layers[0])
        self.fc2=nn.Linear(layers[0], layers[1])
        self.fc3=nn.Linear(layers[1], out_sz)
        
    def forward(self, X):
        X=F.relu(self.fc1(X))
        X=F.relu(self.fc2(X))
        X=self.fc3(X)
        return F.log_softmax(X, dim=1)
    
torch.manual_seed(101)
model=MultilayerPerceptron()

def count_parameters(model):
    params=[p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f"{item:>6}")
    print(f"______\n{sum(params):>6}")
    
#print(count_parameters(model))

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=0.001)

for images, labels in train_load:
    print("Batch Shape: ", images.size())
    break

#print(images.view(100,-1).size())

start_time= time.time()
epochs=10
train_losses=[]
test_losses=[]
train_correct=[]
test_correct=[]

for i in range(epochs):
    train_corr=0
    test_corr=0
    
    for b, (X_train, y_train) in enumerate(train_load):
        b+=1
        
        y_pred=model(X_train.view(100,-1))
        loss=criterion(y_pred,y_train)
        
        predicted=torch.max(y_pred.data, 1)[1]
        batch_corr=(predicted==y_train).sum()
        train_corr+=batch_corr
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if b%200==0:
            print(f"Epochs: {i:2} Batch: {b:4} [{100*b:6}/60000] Loss: {loss.item():10.8f} Train Accuracy: {train_corr.item()*100/(100*b):7.3f}%")
            
    train_losses.append(loss)
    train_correct.append(train_corr)
    
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_load):
            y_val=model(X_test.view(10000,-1))
            
            predicted=torch.max(y_val.data,1)[1]
            test_corr+=(predicted==y_test).sum()
            
    loss=criterion(y_val,y_test)
    test_losses.append(loss)
    test_correct.append(test_corr)
    
print(f"Test Accuracy: {test_correct[-1]*100/10000:7.3f}%")
print(f"\nDuration: {time.time()-start_time:.0f} seconds")

plt.subplot(2,1,1)
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss For Each Epoch")
plt.legend()

plt.subplot(2,1,2)
plt.plot([t/600 for t in train_correct], label="Training Accuracy")
plt.plot([t/100 for t in test_correct], label="Test Accuracy")
plt.title("Accuracy For Each Epoch")
plt.legend()

plt.tight_layout()
plt.show()

np.set_printoptions(formatter=dict(int=lambda x:f"{x:4}"))
print(np.arange(10).reshape(1,10))
print("\n", confusion_matrix(predicted.view(-1), y_test.view(-1)))

misses=np.array([])
for i in range(len(predicted.view(-1))):
    if predicted[i]!=y_test[i]:
        misses=np.append(misses,i).astype("int64")

misses[0:-1].reshape(1,-1)

r=12
row=iter(np.array_split(misses,len(misses)//r+1))

nextrow=next(row)
print("index: ", nextrow)
print("Label: ", y_test.index_select(0,torch.tensor(nextrow)).numpy())
print("Guess:", predicted.index_select(0,torch.tensor(nextrow)).numpy())

images=X_test.index_select(0,torch.tensor(nextrow))
im=make_grid(images, nrow=r)
plt.figure(figsize=(10,4))
plt.imshow(np.transpose(im.numpy(),(1,2,0)))
plt.show()