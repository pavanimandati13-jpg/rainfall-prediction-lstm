"""
=========================================================================
 LOAD SAVED MODEL AND MAKE NEW PREDICTIONS
=========================================================================
This script shows how to load the LSTM model AFTER it has already been
trained and saved by rainfall_lstm.py (Step 19), without re-training it.

This is exactly what you would do in a real application: train once,
then reuse the saved model many times.

Run with:
    python load_and_predict.py
(Run rainfall_lstm.py at least once first, so that the saved_model/
 folder actually contains rainfall_lstm_model.pth and scaler.save)
=========================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import joblib

MODEL_PATH = os.path.join("saved_model", "rainfall_lstm_model.pth")
SCALER_PATH = os.path.join("saved_model", "scaler.save")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------
# The model CLASS definition must be available before we can load
# the saved weights into it (PyTorch needs to know the architecture).
# This is an exact copy of the class from rainfall_lstm.py.
# ---------------------------------------------------------------
class RainfallLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(RainfallLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        last_step_out = lstm_out[:, -1, :]
        out = self.dropout(last_step_out)
        out = self.fc(out)
        return out


def load_trained_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Saved model or scaler not found. Run 'python rainfall_lstm.py' "
            "first to train and save them."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = RainfallLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # IMPORTANT: switches off dropout for inference

    scaler = joblib.load(SCALER_PATH)

    metadata = {
        "sequence_length": checkpoint["sequence_length"],
        "feature_columns": checkpoint["feature_columns"],
        "target_column": checkpoint["target_column"],
    }
    return model, scaler, metadata


def predict_next_day_rainfall(recent_days_data, model, scaler, metadata):
    """
    recent_days_data: list of [Temperature, Humidity, Pressure, WindSpeed, Rainfall]
    for the last `sequence_length` days, oldest first.
    """
    feature_columns = metadata["feature_columns"]
    seq_len = metadata["sequence_length"]
    target_idx = feature_columns.index(metadata["target_column"])

    recent_array = np.array(recent_days_data, dtype=float)
    if recent_array.shape != (seq_len, len(feature_columns)):
        raise ValueError(
            f"Expected shape ({seq_len}, {len(feature_columns)}), got {recent_array.shape}"
        )

    scaled_input = scaler.transform(recent_array)
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        scaled_pred = model(input_tensor).cpu().numpy().flatten()

    dummy = np.zeros((1, len(feature_columns)))
    dummy[:, target_idx] = scaled_pred
    prediction = scaler.inverse_transform(dummy)[:, target_idx][0]
    return max(0.0, float(prediction))


if __name__ == "__main__":
    model, scaler, metadata = load_trained_model()
    print("Model loaded successfully.")
    print("Expected input: last", metadata["sequence_length"], "days of",
          metadata["feature_columns"])

    # ---- Example usage with made-up values (replace with real recent data) ----
    example_days = [
        [26.5, 70, 1011, 12, 3.2],
        [27.0, 72, 1010, 13, 5.0],
        [26.8, 75, 1009, 14, 8.1],
        [25.9, 78, 1008, 15, 12.4],
        [25.2, 80, 1007, 16, 15.0],
        [26.0, 76, 1009, 14, 6.2],
        [26.7, 73, 1010, 13, 2.0],
        [27.3, 68, 1012, 11, 0.0],
        [27.9, 65, 1013, 10, 0.0],
        [28.1, 63, 1014, 9, 0.0],
        [27.5, 66, 1013, 10, 0.5],
        [27.0, 70, 1011, 12, 4.0],
        [26.4, 74, 1010, 13, 7.5],
        [25.8, 79, 1008, 15, 11.0],
    ]  # Must have exactly `sequence_length` rows (14 by default)

    result = predict_next_day_rainfall(example_days, model, scaler, metadata)
    print(f"\nPredicted rainfall for the next day: {result:.2f} mm")
