import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🌧️",
    layout="wide"
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

class RainfallLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64,
                 num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]

        output = self.dropout(last_hidden)

        return self.fc(output)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = "saved_model/rainfall_lstm_model.pth"
SCALER_PATH = "saved_model/scaler.save"

FEATURES = [
    "Temperature",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "Rainfall"
]


@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        model = RainfallLSTM()

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model = RainfallLSTM()

        model.load_state_dict(checkpoint)

    model.eval()

    scaler = joblib.load(SCALER_PATH)

    return model, scaler


try:

    model, scaler = load_model()

except Exception as e:

    st.error("Could not load the trained model.")

    st.code(str(e))

    st.stop()


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f9ff;
    }

    .title {
        text-align: center;
        color: #1261a0;
        font-size: 42px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result {
        background: linear-gradient(
            135deg,
            #1261a0,
            #29a9df
        );

        padding: 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-top: 30px;
    }

    .result-number {
        font-size: 50px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">🌧️ Rainfall Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'LSTM Deep Learning Weather Forecasting System'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "⚡ Powered by PyTorch LSTM | "
    "Prediction based on the previous 14 days"
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("📋 Project Information")

st.sidebar.write(
    """
    **Model:** 2-Layer LSTM

    **Framework:** PyTorch

    **Sequence Length:** 14 Days

    **Features:** 5

    **Prediction:** Next-Day Rainfall
    """
)

st.sidebar.markdown("---")

st.sidebar.write("### 📊 Model Performance")

st.sidebar.metric(
    "MAE",
    "5.109 mm"
)

st.sidebar.metric(
    "RMSE",
    "7.382 mm"
)

st.sidebar.metric(
    "R² Score",
    "0.234"
)


# --------------------------------------------------
# INPUT
# --------------------------------------------------

st.header("📊 Enter Weather Data")

st.write(
    "Enter weather observations for the previous "
    "**14 days**."
)

# Sample data button
if st.button("🔄 Fill Sample Data"):

    st.session_state.sample = True

else:

    if "sample" not in st.session_state:
        st.session_state.sample = True


# --------------------------------------------------
# DATA INPUT
# --------------------------------------------------

data = []

for day in range(14):

    st.markdown(f"### Day {day + 1}")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        temperature = st.number_input(
            "Temperature (°C)",
            value=25.0,
            step=0.1,
            key=f"temp_{day}"
        )

    with col2:

        humidity = st.number_input(
            "Humidity (%)",
            value=60.0,
            step=0.1,
            key=f"humidity_{day}"
        )

    with col3:

        pressure = st.number_input(
            "Pressure (hPa)",
            value=1013.0,
            step=0.1,
            key=f"pressure_{day}"
        )

    with col4:

        windspeed = st.number_input(
            "Wind Speed",
            value=12.0,
            step=0.1,
            key=f"wind_{day}"
        )

    with col5:

        rainfall = st.number_input(
            "Rainfall (mm)",
            value=0.0,
            min_value=0.0,
            step=0.1,
            key=f"rain_{day}"
        )

    data.append([
        temperature,
        humidity,
        pressure,
        windspeed,
        rainfall
    ])


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.markdown("---")

if st.button(
    "🔮 Predict Next-Day Rainfall",
    type="primary",
    use_container_width=True
):

    try:

        # Convert input to numpy
        input_data = np.array(
            data,
            dtype=np.float32
        )

        # Scale using training scaler
        scaled_data = scaler.transform(
            input_data
        )

        # Convert to PyTorch tensor
        X = torch.tensor(
            scaled_data,
            dtype=torch.float32
        )

        # Add batch dimension
        X = X.unsqueeze(0)

        # Prediction
        with torch.no_grad():

            prediction_scaled = model(X)

        prediction_scaled = (
            prediction_scaled
            .cpu()
            .numpy()
            .reshape(-1, 1)
        )

        # Inverse scaling
        dummy = np.zeros(
            (1, 5),
            dtype=np.float32
        )

        dummy[0, 4] = prediction_scaled[0, 0]

        prediction = scaler.inverse_transform(
            dummy
        )[0, 4]

        prediction = max(
            0,
            float(prediction)
        )

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        if prediction < 1:

            icon = "☀️"
            status = "Little or No Rain Expected"

        elif prediction < 10:

            icon = "🌦️"
            status = "Light Rain Expected"

        elif prediction < 25:

            icon = "🌧️"
            status = "Moderate Rain Expected"

        else:

            icon = "⛈️"
            status = "Heavy Rain Expected"


        st.markdown(
            f"""
            <div class="result">

                <div style="font-size:60px;">
                    {icon}
                </div>

                <h2>Next-Day Rainfall</h2>

                <div class="result-number">
                    {prediction:.2f} mm
                </div>

                <h3>{status}</h3>

                <p>
                    Prediction generated using
                    your trained PyTorch LSTM model.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Rainfall Prediction System • "
    "Deep Learning Project • "
    "Python + PyTorch + Streamlit"
)
