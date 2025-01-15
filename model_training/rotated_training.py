import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

#hadron_level_data = pd.read_csv("./hadron_level_events_rotated_half_pi.csv", header = 0)
#parton_level_data = pd.read_csv("./parton_level_events_rotated_half_pi.csv", header = 0)
hadron_level_data = pd.read_csv("./hadron_level_events.csv")
parton_level_data = pd.read_csv("./parton_level_events.csv")

original = pd.read_csv("./hadron_level_events.csv")

print(parton_level_data)
print(hadron_level_data)
print(original)

parton_level_data = parton_level_data.drop(columns=["Unnamed: 61"]).dropna()
hadron_level_data = hadron_level_data.drop(columns=["Unnamed: 91"]).dropna()
original = original.drop(columns=["Unnamed: 91"]).dropna()

#parton_level_data = parton_level_data.drop(columns=["Unnamed: 0"]).dropna()
#hadron_level_data = hadron_level_data.drop(columns=["Unnamed: 0"]).dropna()

print(parton_level_data)
print(hadron_level_data)
print(original)


x = hadron_level_data.values
y = parton_level_data.values
z = original.values

scx = StandardScaler()
x = scx.fit_transform(x)

scy = StandardScaler()
y = scy.fit_transform(y)

class dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=21, shuffle=False)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')

X_train = torch.from_numpy(X_train).to(device)
X_test = torch.from_numpy(X_test).to(device)
Y_train = torch.from_numpy(Y_train).to(device)
Y_test = torch.from_numpy(Y_test).to(device)


train_set = dataset(X_train, Y_train)
train_loader = DataLoader(train_set, batch_size=16,shuffle=False)

test_set = dataset(X_test, Y_test)

class Net(nn.Module):
    def __init__(self, input_shape):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 61)

    def forward(self,x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x).relu()
        x = self.fc4(x).relu()
        x = self.fc5(x)
        return x

lr = 0.0001
model = Net(input_shape=x.shape[1])
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# loss_fn = MSELoss()
loss_fn = nn.MSELoss()

model.load_state_dict(torch.load("./model.pth", weights_only=True, map_location=torch.device('cpu')))

epochs = 3000
losses = []
accur = []
accur_test = []
for i in tqdm(range(epochs + 1)):
  for j,(x_train,y_train) in enumerate(train_loader):
    output = model(x_train)
    # print(output.shape)
    # print(y_train.shape)
    # print(output[0].shape)
    # print(output[0].shape)
    loss = loss_fn(output,y_train)


    # predicted = model(X_train)
    # predicted = predicted.reshape(-1).cpu().detach().numpy().round()
    # acc = (predicted == Y_train.cpu().numpy()).mean()

    # predicted_test = model(X_test)
    # predicted_test = predicted_test.reshape(-1).cpu().detach().numpy().round()
    # acc_test = (predicted_test == Y_test.cpu().numpy()).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if i % 10 == 0:
  losses.append(loss.item())
  # accur.append(acc)
  # accur_test.append(acc_test)
  print("Epoch: {} \t Loss: {:.6f}".format(i,loss))
  # print("Epoch: {} \t Loss: {:.6f}\t Train Accuracy: {:.6f}\t Test Accuracy: {:.6f}".format(i,loss,acc, acc_test))



torch.save(model.state_dict(), './model_nonrotated.pth')
