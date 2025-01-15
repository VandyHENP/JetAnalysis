# Author: Michael and Umar

import numpy as np
import json
import copy
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import energyflow as ef


print("Testing")

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

def load_graphs(file_path):
    with open(file_path, 'r') as file:
        graphs = json.load(file)
    return graphs


hadron_level_data = pd.read_csv("hadron_level_events.csv")
parton_level_data = pd.read_csv("parton_level_events.csv")
parton_events = pd.read_csv("parton_level_events.csv")


hadron_level_data = hadron_level_data.drop(columns=["Unnamed: 91"]).dropna()
parton_level_data = parton_level_data.drop(columns=["Unnamed: 61"]).dropna()
parton_events = parton_events.drop(columns=["Unnamed: 61"]).dropna()

hadron_level_data_rotated = pd.read_csv("hadron_level_events_rotated_half_pi.csv")
parton_level_data_rotated = pd.read_csv("parton_level_events_rotated_half_pi.csv")

parton_level_data_rotated = parton_level_data_rotated.drop(columns=["Unnamed: 0"]).dropna()
hadron_level_data_rotated = hadron_level_data_rotated.drop(columns=["Unnamed: 0"]).dropna()

x = hadron_level_data.values
x_rotated = hadron_level_data_rotated.values

y = parton_level_data.values
y_rotated = parton_level_data_rotated.values

partons = parton_events.values

scx = StandardScaler()
x = scx.fit_transform(x)

scx_rotated = StandardScaler()
x_rotated = scx_rotated.fit_transform(x_rotated)
scy = StandardScaler()
y = scy.fit_transform(y)

scy_rotated = StandardScaler()
y_rotated = scy_rotated.fit_transform(y_rotated)

print("Printing after SS")
print("Printing x")
print(x[0])
print(x_rotated[0])
device = "cuda" if torch.cuda.is_available() else "cpu"


# Don't think i need the next to lines, but not sure
_, X_test_dummy, _, _ = train_test_split(x, y, test_size=0.1, random_state=21, shuffle=False)
_, X_test_dummy_rotated, _, _ = train_test_split(x_rotated, y_rotated, test_size=0.1, random_state=21, shuffle=False)

X_test = x
X_test_rotated = x_rotated

X_test_rotated = X_test_rotated.astype('float32')
X_test = X_test.astype('float32')

X_test_rotated = torch.from_numpy(X_test_rotated).to(device)
X_test = torch.from_numpy(X_test).to(device)


# Loads the models
model_nonrotated = Net(input_shape=x.shape[1])
model_nonrotated.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))

model_rotated = Net(input_shape=x.shape[1])
model_rotated.load_state_dict(torch.load('model_rotated.pth', map_location=torch.device('cpu')))

# Printing rotated and non-rotated for sanity check
print("Printing xtest")
print("Printing not rotated")
print(X_test[0])
print("Printing rotated")
print(X_test_rotated[0])

predicted_rotated = model_rotated(X_test)
predicted_nonrotated = model_nonrotated(X_test)

predicted_rotated_rotatedset = model_rotated(X_test_rotated)
predicted_nonrotated_rotatedset = model_nonrotated(X_test_rotated)

predicted_rotated = scy.inverse_transform(predicted_rotated.detach().cpu().numpy())
predicted_nonrotated = scy.inverse_transform(predicted_nonrotated.detach().cpu().numpy())

predicted_rotated_rotatedset = scy_rotated.inverse_transform(predicted_rotated_rotatedset.detach().cpu().numpy())
predicted_nonrotated_rotatedset = scy_rotated.inverse_transform(predicted_nonrotated_rotatedset.detach().cpu().numpy())

# Analysing first event
ev0_rotated = predicted_rotated[0][1:].reshape(20,3)
ev0_nonrotated = predicted_nonrotated[0][1:].reshape(20,3)

ev0_rotated_rotatedset = predicted_rotated_rotatedset[0][1:].reshape(20,3)
ev0_nonrotated_rotatedset = predicted_nonrotated_rotatedset[0][1:].reshape(20,3)


# analysing different event
event = 7 # which event we are looking at right now in the last two printed dataset
ev1_rotated = predicted_rotated[event][1:].reshape(20,3)
ev1_nonrotated = predicted_nonrotated[event][1:].reshape(20,3)


ev1_rotated_rotatedset = predicted_rotated_rotatedset[1][1:].reshape(20,3)
ev1_nonrotated_rotatedset = predicted_nonrotated_rotatedset[1][1:].reshape(20,3)

ev0_rotated_rotatedset = np.array(ev0_rotated_rotatedset)
ev0_nonrotated_rotatedset = np.array(ev0_nonrotated_rotatedset)


ev1_rotated_rotatedset = np.array(ev1_rotated_rotatedset)
ev1_nonrotated_rotatedset = np.array(ev1_nonrotated_rotatedset)

print("Printing predictions on the non rotated dataset")
print("nonrotated model:")
print(ev0_nonrotated)
print("rotated model:")
print(ev0_rotated)

print("Printing predictions on the rotated dataset")
print("nonrotated model:")
print(ev0_nonrotated_rotatedset)
print("rotated model:")
print(ev0_rotated_rotatedset)

print()
print("ANALYSING OTHER EVENT")
print()

print("Printing predictions on the non rotated dataset")
print("nonrotated model")
print(ev1_nonrotated)
print("rotated model")
print(ev1_rotated)

print("Printing predictions on the rotated dataset")
print("nonrotated modl")
print(ev1_nonrotated_rotatedset)
print("rotated model")
print(ev1_rotated_rotatedset)

# The model producs insignificantly small negative pT values
# Reset them to zero here to calculate the EMD
ev1_nonrotated = ev1_nonrotated[ev1_nonrotated[:, 0] > 0]
ev1_rotated = ev1_rotated[ev1_rotated[:, 0] > 0]

ev0_rotated_rotatedset = ev0_rotated_rotatedset[ev0_rotated_rotatedset[:, 0] > 0]
ev0_rotated = ev0_rotated[ev0_rotated[:, 0] > 0]


emdval, G = ef.emd.emd(ev1_rotated, ev1_nonrotated, R=1, return_flow=True)
print("printing emdval between the rotated models prediction and the non rotated models predictions")
print(emdval)
print()
emdval, G = ef.emd.emd(ev0_rotated, ev0_rotated_rotatedset, R=1, return_flow=True)
print()
print("printing emdval between the rotated models predictions on the rotated and non rotated set")
print(emdval)

# a = 0
# b = 0
# for parton in ev1_rotated:
# 	a+=parton[2]
#
# for parton in ev1_nonrotated:
# 	b+=parton[2]
# print(a)
# print()
# print(b)


