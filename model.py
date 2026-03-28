# %%
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# %%
OUT_FEATURE = "Call Volume"

call_centers = ["A", "B", "C", "D"]
types = ["Daily", "Interval"]

dfs = pd.read_excel("data/Data for Datathon (Revised).xlsx", sheet_name=[
    f"{center} - {type}"
    for center in call_centers
    for type in types
])

# %%
# df = dfs["A - Interval"].dropna() #[["Date", "Call Volume"]]
center_map = {
    "A" : 0,
    "B" : 1,
    "C" : 2,
    "D" : 3
}
for c in call_centers:
    dfs[f"{c} - Interval"]["center"] = center_map[c]
    
df = pd.concat([dfs[f"{c} - Interval"] for c in call_centers]).dropna()
df

# %%
# plt.plot(df["Interval"], df["Call Volume"])
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
    # print(dt)
    hour.append(dt.hour)
    minute.append(dt.minute)

hour, minute = [np.array(x) for x in (hour, minute)]

plt.plot(day, df["Call Volume"])

# %%
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

out = df[["Call Volume"]]
features =  df[[
    f"{c}-{t}"
    for c in cyclic_columns
    for t in types
] + ["center"]]
# time_features
pd.concat((out, features), axis=1)
# out

# %%
def create_sequences(values, time_features, seq_len):
    X, y = [], []

    for i in range(len(values) - seq_len):
        seq_x = []

        for j in range(seq_len):
            timestep_features = np.concatenate([
                [values[i + j]],          # out
                time_features[i + j]      # time features
            ])
            seq_x.append(timestep_features)

        X.append(seq_x)
        y.append(values[i + seq_len])

    return np.array(X), np.array(y)
    
scaler = MinMaxScaler(feature_range=(-1, 1))
values = scaler.fit_transform(out.to_numpy()).flatten()
features = features.to_numpy()

X, y = create_sequences(values, features, lag := 48 * 7)

# %%
device = 'cuda:0' if torch.cuda.is_available else 'cpu'
device

# %%

X = torch.tensor(X, dtype=torch.bfloat16, device=device)
y = torch.tensor(y, dtype=torch.bfloat16, device=device)
# X = torch.tensor(shifted[:, 1:], dtype=torch.bfloat16, device=device)
# y = torch.tensor(shifted[:, 0], dtype=torch.bfloat16, device=device)

X.shape, y.shape

# %%
split_index = int(len(X) * .95)
split_index

# %%
# need to add dimensions for LSTM
X_train = X[:split_index].reshape(-1, lag, 1 + 9)   # 8 additional time features + 1 centers
X_test = X[split_index:].reshape(-1, lag, 1 + 9)

y_train = y[:split_index].reshape(-1, 1)
y_test = y[split_index:].reshape(-1, 1)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %%
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

# %%
train_dataset, test_dataset

# %%
from torch.utils.data import DataLoader

batch_size = 32 # 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)

# %%
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
        
        self.fc1 = nn.Linear(hidden_size, 32, dtype=torch.bfloat16)     
        self.fc2 = nn.Linear(32, 1, dtype=torch.bfloat16)     
        # fc to map hidden size to just 1, final value

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, dtype=torch.bfloat16).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, dtype=torch.bfloat16).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out

model = LSTM(10, 64, 3)
model.to(device)
model

# %%
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

# %%
from tqdm import tqdm

learning_rate = 0.001
num_epochs = 5
# loss_function = nn.MSELoss()
def quantile_loss(y_pred, y_true, q=0.7):
    error = y_true - y_pred
    return torch.mean(torch.max(q * error, (q - 1) * error))
loss_function = lambda y_pred, y_true: quantile_loss(y_pred, y_true, q=0.7)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(num_epochs)):
    train_one_epoch()
    validate_one_epoch()

# %%
with torch.no_grad():
    predicted = model(X_train).to(torch.float32).cpu().numpy()

plt.plot(scaler.inverse_transform(y_train.to(torch.float32).cpu().numpy()), label=f'Actual {OUT_FEATURE}')
plt.plot(
    scaler.inverse_transform(predicted), 
    label=f'Predicted {OUT_FEATURE}')
plt.xlabel('Date')
plt.ylabel(OUT_FEATURE)
plt.legend()
plt.show()


# %%
with torch.no_grad():
    predicted = model(X_test).to(torch.float32).cpu().numpy()

plt.plot(
    scaler.inverse_transform(y_test.to(torch.float32).cpu().numpy()), 
    label=f'Actual {OUT_FEATURE}')
plt.plot(
    scaler.inverse_transform(predicted), 
    label=f'Predicted {OUT_FEATURE}')
plt.xlabel('Date')
plt.ylabel(OUT_FEATURE)
plt.legend()
plt.show()


# %%
forecast_df = pd.read_csv("data/template_forecast_v00.csv")[["Month", "Day", "Interval"]]
forecast_df

# %%
import numpy as np
import torch
import pandas as pd

# ── 1. Cyclically encode August's time features ─────────────────────────────

def get_august_features(center_id: int) -> np.ndarray:
    """
    Build the 9 time-features (no CCT) for every 30-min interval in August
    for a given call-center id.
    Returns shape (1488, 9).
    """
    rows = []
    for day in range(1, 32):          # August has 31 days
        for hour in range(24):
            for minute in (0, 30):
                month = 8
                rows.append([
                    np.sin(2 * np.pi * month / 12),
                    np.cos(2 * np.pi * month / 12),
                    np.sin(2 * np.pi * day   / 31),
                    np.cos(2 * np.pi * day   / 31),
                    np.sin(2 * np.pi * hour  / 24),
                    np.cos(2 * np.pi * hour  / 24),
                    np.sin(2 * np.pi * minute / 60),
                    np.cos(2 * np.pi * minute / 60),
                    center_id,
                ])
    return np.array(rows, dtype=np.float32)   # (1488, 9)


# ── 2. Iterative forecast for one center ────────────────────────────────────

def forecast_august(model, scaler, values, features, lag, device, center_id):
    """
    values  : scaled CCT values from training data (all of Apr-Jun), shape (N,)
    features: time-features from training data,                       shape (N, 9)
    lag     : look-back window (336)
    Returns  : inverse-transformed CCT predictions for August, shape (1488,)
    """
    model.eval()

    aug_time_features = get_august_features(center_id)  # (1488, 9)
    n_forecast = len(aug_time_features)                  # 1488

    # Seed the sliding window from the last `lag` steps of training data
    # Each row = [CCT_scaled, feat_0, ..., feat_8]
    window_out   = list(values[-lag:])        # length lag, scaled CCT
    window_feats = list(features[-lag:])      # length lag, shape (9,) each

    predictions_scaled = []

    with torch.no_grad():
        for i in range(n_forecast):
            # Build the input sequence: (1, lag, 10)
            seq = np.array([
                np.concatenate([[window_out[j]], window_feats[j]])
                for j in range(lag)
            ], dtype=np.float32)                         # (lag, 10)

            x = torch.tensor(seq, dtype=torch.bfloat16, device=device)
            x = x.unsqueeze(0)                           # (1, lag, 10)

            pred_scaled = model(x).item()                # scalar

            predictions_scaled.append(pred_scaled)

            # Slide the window: drop oldest, append new step
            window_out.pop(0)
            window_out.append(pred_scaled)

            window_feats.pop(0)
            window_feats.append(aug_time_features[i])    # next Aug time-feat

    # Inverse-transform
    preds_array = np.array(predictions_scaled, dtype=np.float32).reshape(-1, 1)
    predictions = scaler.inverse_transform(preds_array).flatten()
    return predictions


# ── 3. Run forecast (for center 0 as an example; loop over all centers) ─────

august_preds = [forecast_august(
    model   = model,
    scaler  = scaler,
    values  = values,          # the scaled CCT array from training
    features= features,        # the (N, 9) time-feature array from training
    lag     = lag,             # 336
    device  = device,
    center_id = center,             # change per center: 0, 1, 2, 3
) for center in range(4)]

# print("August predictions shape:", august_preds.shape)   # (1488,)
# print("First 5 predictions (CCT in seconds):", august_preds[:5])

# %%
forecast_df = pd.read_csv("forecasts/forecast_v11.csv")
forecast_df

# %%
out_format = {
    "CCT" : "CCT",
    "Call Volume" : "Calls_Offered",
    "Abandoned Calls" : "Abandoned_Calls",
    "Abandoned Rate" : "Abandoned_Rate"
}

for i, center in enumerate(("A", "B", "C", "D")):
    forecast_df[f"{out_format[OUT_FEATURE]}_{center}"] = august_preds[i]
forecast_df

# %%
forecast_df.to_csv("forecast_vxx.csv")



# %%
