"""
=========================================================================
 RAINFALL PREDICTION USING LSTM (PyTorch)
=========================================================================
A beginner-friendly, end-to-end deep learning project that predicts the
NEXT DAY's rainfall (in mm) from the previous N days of weather data,
using an LSTM (Long Short-Term Memory) neural network built with PyTorch.

This single script covers the ENTIRE pipeline:
    1. Load data
    2. Explore / analyze data
    3. Clean data (handle missing values)
    4. Preprocess dates & sort chronologically
    5. Select features & normalize
    6. Build time-series sequences
    7. Chronological train/test split
    8. Define the LSTM model (PyTorch nn.LSTM)
    9. Train the model
   10. Evaluate the model (MAE, RMSE, R^2)
   11. Plot results
   12. Predict rainfall from a user's custom input
   13. Save the trained model + scaler for later reuse

Run with:
    python rainfall_lstm.py

No TensorFlow / Keras is used anywhere in this project -- only PyTorch.
=========================================================================
"""

# -------------------------------------------------------------------
# SECTION 0: IMPORTS
# -------------------------------------------------------------------
# pandas / numpy   -> data loading & numerical operations
# scikit-learn     -> scaling data + evaluation metrics
# matplotlib/seaborn -> plotting
# torch            -> building and training the LSTM neural network
# -------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Make results reproducible (same "random" results every run)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Use GPU automatically if available (helpful in Google Colab), else CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------------
# SECTION 1: CONFIGURATION
# -------------------------------------------------------------------
# Keeping all the "tunable" settings in one place makes the project easy
# to understand and experiment with.
# -------------------------------------------------------------------
CSV_PATH = os.path.join("data", "sample_rainfall_data.csv")  # change this to your own CSV
FEATURE_COLUMNS = ["Temperature", "Humidity", "Pressure", "WindSpeed", "Rainfall"]
TARGET_COLUMN = "Rainfall"          # what we want to predict
SEQUENCE_LENGTH = 14                # use the past 14 days to predict day 15
TRAIN_SPLIT_RATIO = 0.8             # 80% train, 20% test (chronological, not random)
BATCH_SIZE = 32
HIDDEN_SIZE = 64                    # number of "memory units" inside the LSTM
NUM_LSTM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
NUM_EPOCHS = 60

MODEL_SAVE_PATH = os.path.join("saved_model", "rainfall_lstm_model.pth")
SCALER_SAVE_PATH = os.path.join("saved_model", "scaler.save")

os.makedirs("saved_model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# -------------------------------------------------------------------
# SECTION 2: LOAD THE DATASET
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 1-2: LOADING AND EXPLORING THE DATASET")
print("=" * 60)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Could not find '{CSV_PATH}'.\n"
        f"Run 'python data/generate_sample_data.py' first, OR place your own "
        f"weather CSV at this path (see README.md for the required format)."
    )

df = pd.read_csv(CSV_PATH)

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset shape (rows, columns):", df.shape)

print("\nColumn data types:")
print(df.dtypes)

print("\nBasic statistics:")
print(df.describe())

print("\nMissing values BEFORE cleaning:")
print(df.isna().sum())


# -------------------------------------------------------------------
# SECTION 3: HANDLE MISSING VALUES
# -------------------------------------------------------------------
# Real-world weather data almost always has a few missing readings
# (sensor downtime, human error, etc.). Since this is a TIME SERIES,
# the safest simple strategy is:
#   1. Forward-fill: copy the last known value forward
#   2. Backward-fill: for any remaining gaps at the very start
# This preserves the smooth day-to-day trend better than filling
# with a global average would.
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: HANDLING MISSING VALUES")
print("=" * 60)

df = df.ffill().bfill()

print("Missing values AFTER cleaning:")
print(df.isna().sum())
assert df.isna().sum().sum() == 0, "There are still missing values left!"


# -------------------------------------------------------------------
# SECTION 4-5: CONVERT DATE COLUMN & SORT CHRONOLOGICALLY
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4-5: FIXING THE DATE COLUMN AND SORTING")
print("=" * 60)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("Date range:", df["Date"].min().date(), "to", df["Date"].max().date())
print("Data is now sorted from oldest to newest (required for time-series).")


# -------------------------------------------------------------------
# QUICK VISUAL EXPLORATION (helps you "see" the data before modeling)
# -------------------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(df["Date"], df["Rainfall"], color="steelblue", linewidth=0.8)
plt.title("Historical Daily Rainfall")
plt.xlabel("Date")
plt.ylabel("Rainfall (mm)")
plt.tight_layout()
plt.savefig(os.path.join("outputs", "01_historical_rainfall.png"), dpi=120)
plt.close()

plt.figure(figsize=(7, 6))
sns.heatmap(df[FEATURE_COLUMNS].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Weather Features")
plt.tight_layout()
plt.savefig(os.path.join("outputs", "02_correlation_heatmap.png"), dpi=120)
plt.close()

print("Saved exploratory plots to the 'outputs/' folder.")


# -------------------------------------------------------------------
# SECTION 6-7: SELECT FEATURES AND NORMALIZE
# -------------------------------------------------------------------
# Neural networks train much better when all input numbers are on a
# similar scale. MinMaxScaler squashes every feature into the range
# [0, 1] using the formula:  x_scaled = (x - min) / (max - min)
#
# IMPORTANT: We fit the scaler ONLY on the concept of "all data" here
# for simplicity (common in beginner tutorials). For a stricter
# real-world pipeline you would fit the scaler on the TRAIN portion
# only, to avoid any "peeking" at test data. This is mentioned again
# in the README under "Limitations".
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6-7: SELECTING FEATURES AND NORMALIZING (MinMaxScaler)")
print("=" * 60)

feature_data = df[FEATURE_COLUMNS].values  # shape: (num_days, num_features)
target_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(feature_data)

print(f"Features used: {FEATURE_COLUMNS}")
print(f"Scaled data shape: {scaled_data.shape}")
print(f"Scaled data range: min={scaled_data.min():.3f}, max={scaled_data.max():.3f}")


# -------------------------------------------------------------------
# SECTION 8: CREATE TIME-SERIES SEQUENCES
# -------------------------------------------------------------------
# An LSTM needs "windows" of consecutive days as input.
# Example with SEQUENCE_LENGTH = 3:
#   Input:  [Day1, Day2, Day3]  ->  Target: Rainfall on Day4
#   Input:  [Day2, Day3, Day4]  ->  Target: Rainfall on Day5
#   ... and so on (this is called a "sliding window").
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"STEP 8: CREATING SEQUENCES (using past {SEQUENCE_LENGTH} days)")
print("=" * 60)


def create_sequences(data, seq_length, target_col_idx):
    """
    Turns a 2D array of shape (num_days, num_features) into:
        X: (num_samples, seq_length, num_features)  -- input windows
        y: (num_samples,)                            -- next-day target value
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i: i + seq_length])
        y.append(data[i + seq_length, target_col_idx])
    return np.array(X), np.array(y)


X, y = create_sequences(scaled_data, SEQUENCE_LENGTH, target_idx)
print(f"X shape: {X.shape}  (samples, sequence_length, num_features)")
print(f"y shape: {y.shape}  (samples,)")


# -------------------------------------------------------------------
# SECTION 9: CHRONOLOGICAL TRAIN / TEST SPLIT
# -------------------------------------------------------------------
# We must NOT shuffle time-series data randomly, because that would let
# the model "see the future" during training. Instead we simply cut the
# data at a point in time: everything before it is training data,
# everything after it is testing data.
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 9: CHRONOLOGICAL TRAIN/TEST SPLIT")
print("=" * 60)

split_idx = int(len(X) * TRAIN_SPLIT_RATIO)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# -------------------------------------------------------------------
# Wrap the arrays in a PyTorch Dataset + DataLoader
# -------------------------------------------------------------------
class RainfallDataset(Dataset):
    """A simple PyTorch Dataset that stores our (X, y) numpy arrays as tensors."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = RainfallDataset(X_train, y_train)
test_dataset = RainfallDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# shuffle=True here only shuffles the ORDER in which training windows are fed
# to the model each epoch -- it does NOT change what's inside each window,
# so it does not break the time-series integrity.
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -------------------------------------------------------------------
# SECTION 10-11: BUILD THE LSTM MODEL (PyTorch)
# -------------------------------------------------------------------
# Architecture:
#   Input  -> (batch, seq_len, num_features)
#   LSTM layer(s) -> learn temporal patterns across the sequence
#   Dropout -> randomly "turns off" some neurons during training to
#              reduce overfitting
#   Fully Connected (Linear) layer -> maps the LSTM's final hidden state
#              to a single predicted rainfall value
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 10-11: BUILDING THE LSTM MODEL")
print("=" * 60)


class RainfallLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(RainfallLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # batch_first=True means our tensors are shaped as
        # (batch_size, sequence_length, num_features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)  # output: 1 value (next-day rainfall)

    def forward(self, x):
        # h0, c0: initial hidden state and cell state (start at zero)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # lstm_out: output at every time step, shape (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x, (h0, c0))

        # We only need the output from the LAST time step, since that
        # summarizes everything the LSTM learned from the whole sequence.
        last_step_out = lstm_out[:, -1, :]

        out = self.dropout(last_step_out)
        out = self.fc(out)  # shape: (batch, 1)
        return out


model = RainfallLSTM(
    input_size=len(FEATURE_COLUMNS),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LSTM_LAYERS,
    dropout=DROPOUT,
).to(device)

print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal trainable parameters: {total_params:,}")


# -------------------------------------------------------------------
# SECTION 12: LOSS FUNCTION AND OPTIMIZER
# -------------------------------------------------------------------
# MSELoss (Mean Squared Error) is the standard choice for predicting a
# continuous number like rainfall in mm.
# Adam optimizer adapts the learning rate automatically and generally
# works well "out of the box" for beginners.
# -------------------------------------------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# -------------------------------------------------------------------
# SECTION 13: TRAINING LOOP (with validation on the test set each epoch)
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 12-13: TRAINING THE MODEL")
print("=" * 60)

train_losses = []
val_losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    # ---- Training phase ----
    model.train()
    running_train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()          # reset gradients from the previous step
        predictions = model(batch_X)   # forward pass
        loss = criterion(predictions, batch_y)
        loss.backward()                # backpropagation: compute gradients
        optimizer.step()               # update the model's weights

        running_train_loss += loss.item() * batch_X.size(0)

    epoch_train_loss = running_train_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)

    # ---- Validation phase (no weight updates, just measuring performance) ----
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            running_val_loss += loss.item() * batch_X.size(0)

    epoch_val_loss = running_val_loss / len(test_dataset)
    val_losses.append(epoch_val_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
              f"Train Loss: {epoch_train_loss:.5f}  |  Val Loss: {epoch_val_loss:.5f}")

print("\nTraining complete.")


# -------------------------------------------------------------------
# SECTION 17a: PLOT TRAINING VS VALIDATION LOSS
# -------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Training Loss")
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (scaled data)")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("outputs", "03_training_loss.png"), dpi=120)
plt.close()
print("Saved loss curve to outputs/03_training_loss.png")


# -------------------------------------------------------------------
# SECTION 14-15: PREDICT ON TEST DATA & CONVERT BACK TO ORIGINAL SCALE
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 14-15: PREDICTING ON TEST DATA")
print("=" * 60)

model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    scaled_predictions = model(X_test_tensor).cpu().numpy().flatten()

scaled_actual = y_test


def inverse_transform_target(scaled_values, scaler, target_col_idx, num_features):
    """
    MinMaxScaler was fit on ALL features together, so to correctly reverse
    the scaling for just the target column, we build a dummy array with
    the same number of columns, put our values into the target column,
    inverse-transform the whole thing, then pull that column back out.
    """
    dummy = np.zeros((len(scaled_values), num_features))
    dummy[:, target_col_idx] = scaled_values
    return scaler.inverse_transform(dummy)[:, target_col_idx]


predicted_rainfall = inverse_transform_target(
    scaled_predictions, scaler, target_idx, len(FEATURE_COLUMNS)
)
actual_rainfall = inverse_transform_target(
    scaled_actual, scaler, target_idx, len(FEATURE_COLUMNS)
)

# Rainfall cannot physically be negative -- clip any tiny negative predictions
predicted_rainfall = np.clip(predicted_rainfall, 0, None)

print("Sample predictions (first 10 test days):")
comparison_df = pd.DataFrame({
    "Actual Rainfall (mm)": actual_rainfall[:10],
    "Predicted Rainfall (mm)": np.round(predicted_rainfall[:10], 2),
})
print(comparison_df)


# -------------------------------------------------------------------
# SECTION 16: EVALUATION METRICS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 16: EVALUATION METRICS")
print("=" * 60)

mae = mean_absolute_error(actual_rainfall, predicted_rainfall)
rmse = np.sqrt(mean_squared_error(actual_rainfall, predicted_rainfall))
r2 = r2_score(actual_rainfall, predicted_rainfall)

print(f"MAE  (Mean Absolute Error):      {mae:.3f} mm")
print(f"RMSE (Root Mean Squared Error):  {rmse:.3f} mm")
print(f"R^2  (R-squared Score):          {r2:.3f}")

print(
    "\nNote: Rainfall is naturally very 'spiky' (many dry days, occasional "
    "heavy bursts), so a modest R^2 is normal and expected -- this is a "
    "genuinely hard forecasting problem, not a bug in the code."
)


# -------------------------------------------------------------------
# SECTION 17b: PLOT ACTUAL VS PREDICTED RAINFALL
# -------------------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(actual_rainfall, label="Actual Rainfall", color="steelblue", linewidth=1.2)
plt.plot(predicted_rainfall, label="Predicted Rainfall", color="orange", linewidth=1.2)
plt.title("Actual vs Predicted Rainfall (Test Set)")
plt.xlabel("Test Sample (chronological)")
plt.ylabel("Rainfall (mm)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("outputs", "04_actual_vs_predicted.png"), dpi=120)
plt.close()
print("Saved actual-vs-predicted plot to outputs/04_actual_vs_predicted.png")


# -------------------------------------------------------------------
# SECTION 18: PREDICT NEXT-DAY RAINFALL FROM USER INPUT
# -------------------------------------------------------------------
# This function lets anyone (e.g. during a live demo) type in the last
# SEQUENCE_LENGTH days of weather observations and get a rainfall
# forecast for the following day.
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 18: CUSTOM PREDICTION FROM USER INPUT")
print("=" * 60)


def predict_next_day_rainfall(recent_days_data, model, scaler, feature_columns, device):
    """
    recent_days_data: a list of lists/tuples, one per day, oldest first, e.g.
        [
            [temp, humidity, pressure, wind, rainfall],   # SEQUENCE_LENGTH days ago
            ...
            [temp, humidity, pressure, wind, rainfall],   # yesterday
        ]
    Returns: predicted rainfall (mm) for the NEXT day.
    """
    recent_array = np.array(recent_days_data, dtype=float)
    if recent_array.shape != (SEQUENCE_LENGTH, len(feature_columns)):
        raise ValueError(
            f"Expected input shape ({SEQUENCE_LENGTH}, {len(feature_columns)}), "
            f"got {recent_array.shape}"
        )

    scaled_input = scaler.transform(recent_array)
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)
    # unsqueeze(0) adds the "batch" dimension -> shape becomes (1, seq_len, num_features)

    model.eval()
    with torch.no_grad():
        scaled_pred = model(input_tensor).cpu().numpy().flatten()

    result = inverse_transform_target(scaled_pred, scaler, target_idx, len(feature_columns))
    return max(0.0, float(result[0]))


# --- Demo: automatically use the LAST `SEQUENCE_LENGTH` real days from the
#     dataset as an example, so the script has a working demo out-of-the-box.
example_recent_days = df[FEATURE_COLUMNS].values[-SEQUENCE_LENGTH:]
demo_prediction = predict_next_day_rainfall(
    example_recent_days, model, scaler, FEATURE_COLUMNS, device
)
print(f"\n[Automatic demo] Using the last {SEQUENCE_LENGTH} days in the dataset,")
print(f"the model predicts NEXT day's rainfall as: {demo_prediction:.2f} mm")

# --- Optional: interactive manual entry (only runs if you execute this file
#     directly and choose to use it; safe to skip in automated/Colab runs).
def run_interactive_prediction():
    print(f"\nEnter the last {SEQUENCE_LENGTH} days of weather data.")
    print(f"For each day, provide: {', '.join(FEATURE_COLUMNS)} (comma-separated).")
    print("Example: 26.5, 70, 1011, 12, 3.2\n")

    manual_days = []
    for day_num in range(1, SEQUENCE_LENGTH + 1):
        while True:
            raw = input(f"Day {day_num}/{SEQUENCE_LENGTH}: ")
            try:
                values = [float(v.strip()) for v in raw.split(",")]
                if len(values) != len(FEATURE_COLUMNS):
                    raise ValueError
                manual_days.append(values)
                break
            except ValueError:
                print(f"  Please enter exactly {len(FEATURE_COLUMNS)} numbers, comma-separated.")

    prediction = predict_next_day_rainfall(manual_days, model, scaler, FEATURE_COLUMNS, device)
    print(f"\nPredicted rainfall for the next day: {prediction:.2f} mm")


if __name__ == "__main__":
    user_choice = os.environ.get("RUN_INTERACTIVE", "no")
    # Set the environment variable RUN_INTERACTIVE=yes before running this
    # script if you want to be prompted for manual keyboard input, e.g.:
    #   RUN_INTERACTIVE=yes python rainfall_lstm.py
    if user_choice.lower() == "yes":
        run_interactive_prediction()


# -------------------------------------------------------------------
# SECTION 19: SAVE THE TRAINED MODEL AND SCALER
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 19: SAVING THE TRAINED MODEL")
print("=" * 60)

torch.save({
    "model_state_dict": model.state_dict(),
    "input_size": len(FEATURE_COLUMNS),
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LSTM_LAYERS,
    "dropout": DROPOUT,
    "sequence_length": SEQUENCE_LENGTH,
    "feature_columns": FEATURE_COLUMNS,
    "target_column": TARGET_COLUMN,
}, MODEL_SAVE_PATH)

import joblib
joblib.dump(scaler, SCALER_SAVE_PATH)

print(f"Model saved to:  {MODEL_SAVE_PATH}")
print(f"Scaler saved to: {SCALER_SAVE_PATH}")

print("\n" + "=" * 60)
print("PROJECT RUN COMPLETE")
print("=" * 60)
print(f"MAE: {mae:.3f} mm | RMSE: {rmse:.3f} mm | R^2: {r2:.3f}")
print("See the 'outputs/' folder for all generated plots.")
print("See load_and_predict.py for how to reuse the saved model later.")
