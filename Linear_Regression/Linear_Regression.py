import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

x_values = torch.linspace(1,50,50).reshape(-1,1)
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

e=torch.randint(-8,9,(50,1),dtype=torch.float)

y_values = 2*x_values+1+e
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out
    
inputDim = 1        
outputDim = 1
epochs = 17
losses=[]
model = linearRegression(inputDim, outputDim)
    
criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(epochs):

    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))

    optimizer.zero_grad()
    outputs = model(inputs)

    loss = criterion(outputs, labels)
    print(loss)
    losses.append(loss)
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))
    
with torch.no_grad(): 
    predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()

plt.plot(range(epochs),losses)
plt.ylabel("MSE LOSS")
plt.xlabel("Epoch")
plt.show()

plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()