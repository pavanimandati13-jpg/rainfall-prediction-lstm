# Rainfall Prediction Using LSTM (PyTorch)

A beginner-friendly, final-year-project-ready deep learning system that predicts
**next-day rainfall (mm)** from recent daily weather observations, using an
**LSTM (Long Short-Term Memory)** neural network built entirely in **PyTorch**
(no TensorFlow, no Keras).

---

## 1. Project Objectives

- Build a deep learning model that learns temporal (day-to-day) patterns in
  weather data to forecast rainfall.
- Demonstrate a complete, real, runnable machine learning pipeline: data
  cleaning → preprocessing → sequence generation → model training →
  evaluation → visualization → deployment-style prediction.
- Keep every step simple and clearly explained so it is suitable as a
  final-year / college demonstration project, while still following
  professional best practices (chronological splitting, saved model
  artifacts, reusable inference script).

---

## 2. Project Abstract (for college submission)

> Rainfall prediction is a critical task in agriculture, disaster management,
> and daily planning, but it is challenging because rainfall depends on
> complex, time-dependent atmospheric interactions. This project presents a
> deep learning approach to short-term rainfall forecasting using a Long
> Short-Term Memory (LSTM) recurrent neural network implemented in PyTorch.
> Historical daily weather data — temperature, humidity, atmospheric
> pressure, wind speed, and rainfall — is cleaned, normalized, and
> transformed into fixed-length time-series sequences representing the
> previous 14 days. These sequences are used to train an LSTM network to
> predict the following day's rainfall amount. The dataset is split
> chronologically into training and testing sets to preserve the temporal
> order of observations, and the model is evaluated using Mean Absolute
> Error (MAE), Root Mean Squared Error (RMSE), and the R² score. The trained
> model is saved for reuse, and a simple interface allows a user to input
> recent weather readings to obtain a next-day rainfall forecast. The
> project demonstrates how recurrent neural networks can capture temporal
> dependencies in meteorological data to support practical, data-driven
> rainfall forecasting.

---

## 3. Project Methodology

1. **Data Acquisition** – Load a CSV of daily weather records (real or the
   provided synthetic dataset).
2. **Exploratory Data Analysis (EDA)** – Inspect shape, types, summary
   statistics, and visualize the rainfall trend and feature correlations.
3. **Data Cleaning** – Handle missing values using forward-fill/back-fill,
   which suits time-ordered data better than replacing gaps with a global
   average.
4. **Date Handling** – Convert the `Date` column to proper `datetime` and
   sort all rows chronologically (oldest → newest).
5. **Feature Selection & Normalization** – Select the numeric weather
   columns and scale them to the [0, 1] range with `MinMaxScaler`, which
   helps the neural network train faster and more stably.
6. **Sequence Construction** – Convert the flat table into overlapping
   sliding windows of `SEQUENCE_LENGTH` (default 14) consecutive days, each
   paired with the rainfall value of the day immediately after the window.
7. **Chronological Train/Test Split** – The first 80% of sequences (in
   time) are used for training, the last 20% for testing — never shuffled
   randomly, since that would leak future information into training.
8. **Model Design** – A 2-layer `nn.LSTM` followed by `Dropout` and a
   `Linear` output layer, implemented in PyTorch.
9. **Training** – Optimize Mean Squared Error loss with the Adam optimizer
   over multiple epochs, tracking both training and validation loss.
10. **Evaluation** – Convert predictions back to millimeters and compute
    MAE, RMSE, and R².
11. **Visualization** – Plot the loss curves and an actual-vs-predicted
    rainfall chart.
12. **Deployment-style Inference** – Provide a function (and an optional
    interactive CLI prompt) that accepts the last 14 days of weather data
    and returns a next-day rainfall forecast, using either the in-memory
    model or a model reloaded from disk.

---

## 4. How LSTM Works (Explanation)

A standard neural network treats every input as independent — it has no
concept of "what came before." Weather, however, is sequential: today's
rainfall is influenced by the last several days of humidity, pressure, and
temperature trends. **Recurrent Neural Networks (RNNs)** were designed to
handle exactly this kind of sequential data by passing information from one
time step to the next.

Plain RNNs struggle to remember information over long sequences (the
"vanishing gradient problem"). **LSTM (Long Short-Term Memory)** networks
solve this using a more sophisticated internal structure called a **cell
state**, which acts like a conveyor belt that carries relevant information
across many time steps, combined with three **gates**:

- **Forget Gate** – decides what old information to discard from the cell
  state (e.g., "yesterday's temperature is no longer very relevant").
- **Input Gate** – decides what new information from the current time step
  to add to the cell state (e.g., "today's rising humidity is important").
- **Output Gate** – decides what part of the cell state to expose as the
  output/hidden state for this time step, which is also passed to the next
  step.

Because of these gates, an LSTM can learn to **remember long-term patterns**
(e.g., a multi-day build-up of humidity before a rainy spell) while also
reacting to **short-term changes**, which makes it well suited to weather
forecasting.

In this project, the LSTM reads 14 days of weather data one day at a time,
updating its internal memory at each step. After processing all 14 days,
the final hidden state — which now summarizes the whole 2-week pattern — is
passed through a small `Linear` layer that outputs a single number: the
predicted rainfall for day 15.

---

## 5. Folder / Project Structure

```
rainfall_lstm_project/
│
├── data/
│   ├── generate_sample_data.py     # Creates a synthetic weather CSV (run first if you have no dataset)
│   └── sample_rainfall_data.csv    # The generated example dataset (4 years of daily data)
│
├── saved_model/                    # Created automatically after training
│   ├── rainfall_lstm_model.pth     # Saved PyTorch model weights + config
│   └── scaler.save                 # Saved MinMaxScaler (needed to reuse the model correctly)
│
├── outputs/                        # Created automatically after running the main script
│   ├── 01_historical_rainfall.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_training_loss.png
│   └── 04_actual_vs_predicted.png
│
├── rainfall_lstm.py                # MAIN SCRIPT: full pipeline (train, evaluate, plot, predict, save)
├── load_and_predict.py             # Loads the saved model and makes a new prediction (no retraining)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 6. Installation

Requires **Python 3.9+**. Works in Google Colab or a local environment.

```bash
# 1. (Optional but recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

If you are on Google Colab, PyTorch is usually pre-installed. If not:

```bash
!pip install torch pandas numpy scikit-learn matplotlib seaborn joblib
```

---

## 7. Dataset

### 7.1 Using the provided synthetic dataset (recommended to start)

If you don't have a real dataset yet, generate one:

```bash
cd data
python generate_sample_data.py
cd ..
```

This creates `data/sample_rainfall_data.csv` — 4 years of realistic,
seasonally-patterned synthetic daily weather data (with a few missing
values deliberately included, so the cleaning step has something to do).

### 7.2 Required CSV format

Your CSV (real or synthetic) must contain these columns (names must match,
case-sensitive, or you must edit `FEATURE_COLUMNS` in `rainfall_lstm.py`):

| Date       | Temperature | Humidity | Pressure | WindSpeed | Rainfall |
|------------|-------------|----------|----------|-----------|----------|
| 2020-01-01 | 17.9        | 39.8     | 1017.4   | 8.4       | 0.0      |
| 2020-01-02 | 17.0        | 34.4     | 1020.3   | 8.5       | 1.2      |
| ...        | ...         | ...      | ...      | ...       | ...      |

- **Date** – any format `pandas.to_datetime()` can parse (e.g. `YYYY-MM-DD`)
- **Temperature** – degrees Celsius
- **Humidity** – percentage (0–100)
- **Pressure** – hectopascals (hPa)
- **WindSpeed** – km/h (or your unit of choice, just be consistent)
- **Rainfall** – millimeters (mm) — this is the value we predict

### 7.3 Using a real public dataset instead

You can replace the CSV with a real dataset, for example:

- **Kaggle** – search "Rainfall Prediction Dataset" or "Weather Dataset"
  (e.g. "Rain in Australia" dataset, or any daily weather CSV with the
  columns above).
- **NOAA Climate Data Online** – https://www.ncdc.noaa.gov/cdo-web/
- **data.gov.in** (India Meteorological Department) – historical rainfall
  and weather records by station.
- **Open-Meteo Historical Weather API** – https://open-meteo.com/ (free,
  no API key required for historical data).

Just make sure the final CSV has (or is renamed to have) the same column
names as above, then point `CSV_PATH` in `rainfall_lstm.py` at your file.

---

## 8. Running the Project

```bash
# Step 1: generate the example dataset (skip if you already have your own CSV)
python data/generate_sample_data.py

# Step 2: run the full training + evaluation pipeline
python rainfall_lstm.py
```

This will print dataset info, cleaning results, training progress every 5
epochs, evaluation metrics, a sample prediction, and save:
- Plots to `outputs/`
- The trained model + scaler to `saved_model/`

### Making a next-day prediction interactively

The main script already prints an automatic demo prediction using the last
14 real days in the dataset. If you want to type in your own 14 days of
weather values from the keyboard:

```bash
RUN_INTERACTIVE=yes python rainfall_lstm.py
```

### Reusing the saved model later (no retraining)

```bash
python load_and_predict.py
```

This loads `saved_model/rainfall_lstm_model.pth` and `saved_model/scaler.save`
and produces a prediction from an example set of 14 days (edit the
`example_days` list in that file with your own real recent readings).

---

## 9. Model Architecture Summary

```
Input:  (batch_size, 14 days, 5 features)
   │
   ▼
LSTM Layer 1  (hidden_size = 64)
   │
   ▼
LSTM Layer 2  (hidden_size = 64)   ← dropout applied between LSTM layers
   │
   ▼
Take output of the LAST time step
   │
   ▼
Dropout (p = 0.2)
   │
   ▼
Fully Connected (Linear) layer  →  1 output value (predicted rainfall, mm)
```

- **Loss function**: Mean Squared Error (`nn.MSELoss`)
- **Optimizer**: Adam (`lr = 0.001`)
- **Epochs**: 60 (configurable)

---

## 10. Advantages

- Captures temporal/sequential patterns in weather data that simple
  regression models (e.g., plain Linear Regression) would miss.
- Fully self-contained: works out of the box with an included synthetic
  dataset, no external downloads required.
- Clear, heavily-commented code suitable for learning and for explaining
  in a viva/demo.
- Modular: swapping in a real dataset only requires changing one file path.
- Model and scaler are saved together, so predictions after reloading are
  guaranteed to use the same normalization as training.

## 11. Limitations

- Rainfall is highly stochastic (random, bursty) by nature; even a
  well-trained model will have real forecasting error — this is expected
  and normal, not a flaw in the implementation.
- The demo dataset is **synthetic**; results on a real dataset (with
  real, messier patterns) will differ and typically require more tuning.
- The scaler in this beginner version is fit on the entire dataset rather
  than strictly on the training portion only, for simplicity — a small
  amount of information about the test set's numeric range "leaks" into
  the scaler. For rigorous research work, fit the scaler on training data
  only, then apply `.transform()` (not `.fit_transform()`) to the test set.
- Only weather variables are used; real rainfall also depends on factors
  like geography, season indicators, satellite data, or nearby station
  readings, which are not included here.
- A single global model is trained; it does not account for different
  climate zones/stations without retraining per-location.

## 12. Future Enhancements

- Fit the `MinMaxScaler` on the training set only (stricter methodology).
- Add more weather features (cloud cover, solar radiation, dew point).
- Try a **GRU** or **Bidirectional LSTM** and compare performance.
- Add **attention mechanisms** to let the model focus on the most relevant
  past days.
- Perform **hyperparameter tuning** (sequence length, hidden size, number
  of layers, learning rate) using grid search or Optuna.
- Turn rainfall prediction into a **classification** problem too (e.g.,
  "will it rain tomorrow: yes/no") alongside the regression forecast.
- Deploy the model behind a simple web app (Flask/FastAPI + a small HTML
  form) for live predictions.
- Train on multiple weather stations/cities and add location as a feature.

---

## 13. Common Errors and Solutions

| Error / Symptom | Likely Cause | Solution |
|---|---|---|
| `FileNotFoundError: Could not find 'data/sample_rainfall_data.csv'` | You haven't generated the sample dataset yet | Run `python data/generate_sample_data.py` first, or set `CSV_PATH` to your own file |
| `ModuleNotFoundError: No module named 'torch'` | PyTorch isn't installed | Run `pip install torch` (or `pip install -r requirements.txt`) |
| `ValueError: could not convert string to float` while training | The `Date` column (or another non-numeric column) accidentally got included in `FEATURE_COLUMNS` | Make sure `FEATURE_COLUMNS` only lists numeric weather columns, not `Date` |
| Loss becomes `nan` during training | Learning rate too high, or unscaled/very large input values | Lower `LEARNING_RATE` (e.g. to 0.0001), double-check that `MinMaxScaler` was actually applied |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Mismatch between the model's expected `input_size` and the actual number of feature columns | Ensure `len(FEATURE_COLUMNS)` matches what the model was built/loaded with |
| `ValueError: Expected input shape (14, 5), got (...)` in a prediction function | You didn't supply exactly `SEQUENCE_LENGTH` days, or missed a feature column | Provide exactly 14 rows, each with all 5 features in the correct order |
| Predictions are all very close to 0 | Very common with rainfall data since most days have little/no rain (class imbalance) | This is expected; consider evaluating with metrics tolerant of skew, or a rain/no-rain classifier as a future enhancement |
| Plots don't display in a plain terminal | `plt.show()` needs a GUI backend, or you're running headless | This project saves plots as PNG files in `outputs/` instead, so no display is required — just open the PNG files |
| `CUDA out of memory` (on Colab GPU) | Batch size too large for the GPU | Lower `BATCH_SIZE`, or switch runtime to CPU for this small dataset (it's fast enough on CPU too) |
| Model gives different results each run | Randomness in weight initialization / data shuffling | A random seed is already set (`SEED = 42`) for reproducibility; results should match across runs on the same machine/library versions |

---

## 14. License / Academic Use

This project is provided as an educational template. Feel free to adapt it,
extend it, and use it as the basis for a college project or dissertation,
citing the public dataset source you ultimately use for real data.
