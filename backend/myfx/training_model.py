
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from .models import ForexData
from .utils import fetch_xauusd

MODEL_PATH = "xauusd_nn.pt"

class ForexNN(nn.Module):
    def __init__(self):
        super(ForexNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def load_data(fetch_if_empty=True, normalize=True):
    """
    Load XAU/USD data from DB.
    - fetch_if_empty: automatically fetch data if table is empty
    - normalize: scale close prices to 0-1 for training
    """
    if ForexData.objects.count() == 0 and fetch_if_empty:
        fetch_xauusd()

    qs = ForexData.objects.all().order_by("date")
    df = pd.DataFrame(list(qs.values("date", "close")))
    if df.empty:
        return None, None, None, None, None

    df["date"] = pd.to_datetime(df["date"])
    
    # Use day index for input
    X = torch.arange(len(df), dtype=torch.float32).unsqueeze(1)
    
    # Normalize close prices
    y = torch.tensor(df["close"].values, dtype=torch.float32).unsqueeze(1)
    if normalize:
        y_min, y_max = y.min(), y.max()
        y_norm = (y - y_min) / (y_max - y_min)
    else:
        y_norm = y
        y_min, y_max = None, None

    return X, y_norm, df, y_min, y_max

def train_nn(epochs=200, lr=0.001):
    X, y, df, _, _ = load_data()
    if X is None:
        print("No data to train")
        return None

    model = ForexNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X)
            val_loss = criterion(val_outputs, y)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {loss.item():.4f}, Eval Loss: {val_loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Neuro Network saved to {MODEL_PATH}")

    return model

def predict_next(days=5, normalize=True):
    X, y, df, y_min, y_max = load_data(fetch_if_empty=True, normalize=normalize)
    if X is None:
        return []

    model = ForexNN()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    last_index = X[-1].item()
    last_date = df["date"].iloc[-1]

    preds = []
    for i in range(1, days + 1):
        future_index = torch.tensor([[last_index + i]], dtype=torch.float32)
        pred = model(future_index)
        if normalize and y_min is not None and y_max is not None:
            pred = pred * (y_max - y_min) + y_min  # denormalize
        preds.append({
            "date": str(last_date + pd.Timedelta(days=i)),
            "predicted_close": round(pred.item(), 2)  # convert tensor to float
        })

    return preds

### version 2
"""
version-2
"""
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from .models import ForexData
# from .utils import fetch_xauusd

# MODEL_PATH = "xauusd_nn.pt"

# class ForexNN(nn.Module):
#     def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
#         super(ForexNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])  # Take the last time step's output
#         return out

# def load_data(fetch_if_empty=True, window_size=30, normalize=True):
#     """
#     Load XAU/USD data from DB and prepare sequences for LSTM.
#     - fetch_if_empty: automatically fetch data if table is empty
#     - window_size: number of past days to use for prediction
#     - normalize: scale close prices to 0-1 for training
#     """
#     if ForexData.objects.count() == 0 and fetch_if_empty:
#         fetch_xauusd()

#     qs = ForexData.objects.all().order_by("date")
#     df = pd.DataFrame(list(qs.values("date", "close")))
#     if df.empty:
#         return None, None, None, None, None

#     df["date"] = pd.to_datetime(df["date"])
    
#     # Normalize close prices
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(df["close"].values.reshape(-1, 1)) if normalize else df["close"].values.reshape(-1, 1)

#     # Create sequences
#     X, y = [], []
#     for i in range(len(scaled_data) - window_size):
#         X.append(scaled_data[i:i + window_size])
#         y.append(scaled_data[i + window_size])

#     X = np.array(X)
#     y = np.array(y)

#     # Convert to tensors
#     X = torch.tensor(X, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.float32)

#     return X, y, df, scaler if normalize else None, window_size

# def train_nn(epochs=200, lr=0.001, train_split=0.8, patience=20):
#     X, y, df, scaler, window_size = load_data()
#     if X is None:
#         print("No data to train")
#         return None

#     # Train-test split (time-series aware)
#     train_size = int(len(X) * train_split)
#     X_train, X_val = X[:train_size], X[train_size:]
#     y_train, y_val = y[:train_size], y[train_size:]

#     model = ForexNN(input_size=1, hidden_size=64, num_layers=2)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # L2 regularization
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

#     best_val_loss = float('inf')
#     early_stop_counter = 0

#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(X_train)
#         loss = criterion(outputs, y_train)
#         loss.backward()
#         optimizer.step()

#         # Validation
#         model.eval()
#         with torch.no_grad():
#             val_outputs = model(X_val)
#             val_loss = criterion(val_outputs, y_val)

#         scheduler.step(val_loss)

#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

#         # Early stopping
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             early_stop_counter = 0
#             torch.save(model.state_dict(), MODEL_PATH)
#         else:
#             early_stop_counter += 1
#             if early_stop_counter >= patience:
#                 print(f"Early stopping at epoch {epoch}")
#                 break

#     print(f"Neuro Network saved to {MODEL_PATH}")
#     return model

# def predict_next(days=5, normalize=True):
#     X, y, df, scaler, window_size = load_data(fetch_if_empty=True, normalize=normalize)
#     if X is None:
#         return []

#     model = ForexNN(input_size=1, hidden_size=64, num_layers=2)
#     model.load_state_dict(torch.load(MODEL_PATH))
#     model.eval()

#     last_date = df["date"].iloc[-1]
#     predictions = []

#     # Start with the last window of actual data
#     current_window = df["close"].values[-window_size:].reshape(-1, 1)
#     if normalize and scaler is not None:
#         current_window = scaler.transform(current_window)
#     current_window = current_window.reshape(1, window_size, 1)
#     current_window = torch.tensor(current_window, dtype=torch.float32)

#     for i in range(days):
#         with torch.no_grad():
#             pred = model(current_window)
#             pred_denorm = scaler.inverse_transform(pred.numpy())[0][0] if normalize and scaler is not None else pred.numpy()[0][0]
#             predictions.append({
#                 "date": str(last_date + pd.Timedelta(days=i + 1)),
#                 "predicted_close": round(pred_denorm, 2)
#             })

#         # Update window with the new prediction (autoregressive)
#         new_pred = pred.numpy().reshape(1, 1, 1)
#         current_window = torch.cat((current_window[:, 1:, :], torch.tensor(new_pred)), dim=1)

#     return predictions