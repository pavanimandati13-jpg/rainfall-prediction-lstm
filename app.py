from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import joblib
import os

app = Flask(__name__)

# ============================================================
# MODEL CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 14
NUM_FEATURES = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

FEATURE_NAMES = [
    "Temperature",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "Rainfall"
]


# ============================================================
# LSTM MODEL
# ============================================================

class RainfallLSTM(nn.Module):

    def __init__(self):
        super(RainfallLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=NUM_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]

        last_hidden = self.dropout(last_hidden)

        output = self.fc(last_hidden)

        return output


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "saved_model/rainfall_lstm_model.pth"
SCALER_PATH = "saved_model/scaler.save"

device = torch.device("cpu")

model = RainfallLSTM()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# The training script may save either a state_dict
# or a checkpoint dictionary.

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

scaler = joblib.load(SCALER_PATH)

print("Model loaded successfully.")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html",
        prediction=None
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Read 14 days of data
        # ----------------------------------------------------

        sequence = []

        for day in range(1, SEQUENCE_LENGTH + 1):

            temperature = float(
                request.form[f"temperature_{day}"]
            )

            humidity = float(
                request.form[f"humidity_{day}"]
            )

            pressure = float(
                request.form[f"pressure_{day}"]
            )

            windspeed = float(
                request.form[f"windspeed_{day}"]
            )

            rainfall = float(
                request.form[f"rainfall_{day}"]
            )

            sequence.append([
                temperature,
                humidity,
                pressure,
                windspeed,
                rainfall
            ])

        # ----------------------------------------------------
        # Convert to numpy
        # ----------------------------------------------------

        sequence = np.array(
            sequence,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Scale input
        # ----------------------------------------------------

        sequence_scaled = scaler.transform(sequence)

        # ----------------------------------------------------
        # Convert to PyTorch tensor
        # ----------------------------------------------------

        X = torch.tensor(
            sequence_scaled,
            dtype=torch.float32
        ).unsqueeze(0)

        # Shape:
        # (1, 14, 5)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            prediction_scaled = model(X)

        prediction_scaled = prediction_scaled.numpy()

        # ----------------------------------------------------
        # Convert rainfall back to original scale
        # ----------------------------------------------------

        # Create dummy array because scaler was fitted
        # on 5 features.

        dummy = np.zeros(
            (1, NUM_FEATURES)
        )

        dummy[:, 4] = prediction_scaled[:, 0]

        prediction_original = scaler.inverse_transform(
            dummy
        )

        rainfall_prediction = float(
            prediction_original[0, 4]
        )

        # Rainfall cannot be negative.

        rainfall_prediction = max(
            0,
            rainfall_prediction
        )

        # ----------------------------------------------------
        # Simple classification
        # ----------------------------------------------------

        if rainfall_prediction < 1:

            status = "Low / No Rainfall"
            icon = "☀️"

        elif rainfall_prediction < 10:

            status = "Light Rain Expected"
            icon = "🌦️"

        elif rainfall_prediction < 25:

            status = "Moderate Rain Expected"
            icon = "🌧️"

        else:

            status = "Heavy Rain Expected"
            icon = "⛈️"

        return render_template(
            "index.html",
            prediction=round(
                rainfall_prediction,
                2
            ),
            status=status,
            icon=icon
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction=None,
            error=str(e)
        )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
