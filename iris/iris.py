import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df=pd.read_csv("iris.csv",sep=",",index_col="Id")
df["Species"]=df["Species"].replace(["Iris-virginica", "Iris-setosa", "Iris-versicolor"], [0,1,2])

class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1=nn.Linear(in_features, h1)
        self.fc2=nn.Linear(h1, h2)
        self.out=nn.Linear(h2, out_features)
        
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.out(x)
        return(x)
    
torch.manual_seed(32)
model=Model()

fig, axes=plt.subplots(nrows=2, ncols=2, figsize=(10,7))
fig.tight_layout()

plots=[(0,1),(2,3),(0,2),(1,3)]
colors=['blue','red','green']
labels=["Iris Setosa", "Iris Virginica", "Iris Versicolor"]

for i, ax in enumerate(axes.flat):
    for j in range(3):
        x=df.columns[plots[i][0]]
        y=df.columns[plots[i][1]]
        ax.scatter(df[df['Species']==j][x], df[df['Species']==j][y], c=colors[j])
        ax.set(xlabel=x, ylabel=y)
        
fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
plt.show()

X=df.drop("Species", axis=1).values
y=df.Species.values

X_train,X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=0.02)

epochs=100
losses=[]

for i in range(epochs):
    y_pred=model.forward(X_train)
    loss=criterion(y_pred, y_train)
    losses.append(loss)
    if i%10==0:
        print(f'Epoch {i} and loss is: {loss.item()}')
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
plt.plot(range(epochs),losses)
plt.ylabel("LOSS")
plt.xlabel("Epoch")
plt.show()

with torch.no_grad():
    y_eval=model.forward(X_test)
    loss=criterion(y_eval, y_test)

print(loss)

correct=0

with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val=model.forward(data)
        print(f'{i+1}. {str(y_val)}   {y_test[i]}')
        
        if y_val.argmax().item()==y_test[i]:
            correct+=1
            
print(f'Dogru sayisi: {correct}')

torch.save(model.state_dict(), "Iris_Modelim.pt")