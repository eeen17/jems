import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

print(f"PyTorch built with CUDA Version: {torch.version.cuda}")

call_centers = ["A", "B", "C", "D"]
types = ["Daily", "Interval"]

dfs = pd.read_excel("data/Data for Datathon (Revised).xlsx", sheet_name=[
    f"{center} - {type}"
    for center in call_centers
    for type in types
])

df = dfs["A - Interval"].dropna() #[["Date", "CCT"]]

# plt.plot(df["Interval"], df["CCT"])
month_map = {
    "April" : 4,
    "May" : 5,
    "June" : 6
}

month = np.array([month_map[m] for m in df["Month"]])

day = df["Day"].to_numpy()

hour = []
minute = []

for dt in df["Interval"]:
    hour.append(dt.hour)
    minute.append(dt.minute)

hour, minute = [np.array(x) for x in (hour, minute)]

plt.plot(day, df["CCT"])

# cyclically encode date
df["month-sin"] = np.sin(2 * np.pi * month / 12)
df["month-cos"] = np.cos(2 * np.pi * month / 12)

df["day-sin"] = np.sin(2 * np.pi * day / 31)
df["day-cos"] = np.cos(2 * np.pi * day / 31)

df["hour-sin"] = np.sin(2 * np.pi * hour / 24)
df["hour-cos"] = np.cos(2 * np.pi * hour / 24)

df["minute-sin"] = np.sin(2 * np.pi * minute / 60)
df["minute-cos"] = np.cos(2 * np.pi * minute / 60)

cyclic_columns = ["month", "day", "hour", "minute"]
types = ["sin", "cos"]

out = df[["CCT"]]
time_features =  df[[
    f"{c}-{t}"
    for c in cyclic_columns
    for t in types
]]
# time_features
pd.concat((out, time_features), axis=1)

def prepare_df_for_lstm(df: pd.DataFrame, n_steps):
    df = dc(df)
    # df['Date'] = pd.to_datetime(df['Date'])
    
    # df.set_index('Date', inplace=True)
    
    for i in reversed(range(1, n_steps+1)):
        df[f'CCT(t-{i})'] = df['CCT'].shift(i)
        
    return df

lag = 48
# shifted_df = prepare_df_for_lstm(out, lag)
# shifted_df.dropna(inplace=True)
shifted_df = pd.concat((prepare_df_for_lstm(out, lag), time_features), axis=1)
shifted_df.dropna(inplace=True)
display(shifted_df)
shifted = shifted_df.to_numpy()

# only scale CCT?
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted = np.hstack((scaler.fit_transform(shifted[:, :lag + 1]), shifted[:, lag + 1:]))
# X = torch.tensor(scaler.fit_transform(X), dtype=torch.bfloat16, device=device)
# y = torch.tensor(y, dtype=torch.bfloat16, device=device)
# np.hstack((shifted, time_features))

shifted.shape

device = 'cuda:0' if torch.cuda.is_available else 'cpu'
device

X = torch.tensor(shifted[:, 1:], dtype=torch.bfloat16, device=device)
y = torch.tensor(shifted[:, 0], dtype=torch.bfloat16, device=device)

X.shape, y.shape

split_index = int(len(X) * .95)

# need to add dimensions for LSTM
X_train = X[:split_index].reshape(-1, lag + 8, 1)   # 8 additional time features
X_test = X[split_index:].reshape(-1, lag + 8, 1)

y_train = y[:split_index].reshape(-1, 1)
y_test = y[split_index:].reshape(-1, 1)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i].to(torch.bfloat16), self.y[i].to(torch.bfloat16)

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

from torch.utils.data import DataLoader

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(                    # batched first -> batch as first dimension
            input_size, 
            hidden_size, 
            num_stacked_layers, 
            batch_first=True, 
            dtype=torch.bfloat16
        )  
        
        self.fc = nn.Linear(hidden_size, 1, dtype=torch.bfloat16)     # fc to map hidden size to just 1, final value

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, dtype=torch.bfloat16).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, dtype=torch.bfloat16).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(1, 4, 1)
model.to(device)
model

def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0], batch[1]

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 10 == 9:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()

def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()


learning_rate = 0.001
num_epochs = 100
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()

with torch.no_grad():
    predicted = model(X_train).to(torch.float32).cpu().numpy()

plt.plot(y_train.to(torch.float32).cpu().numpy(), label='Actual CCT')
plt.plot(predicted, label='Predicted CCT')
plt.xlabel('Date')
plt.ylabel('CCT')
plt.legend()
plt.show()
