# Advanced Time Series Forecasting with Attention-Based Transformers

## 📌 Project Overview
This project implements and evaluates deep learning models for **multivariate time series forecasting** using PyTorch.  
The main goal is to predict **future time steps (multi-step forecasting)** from historical sequential data.

Two models are implemented and compared:

- **Transformer Encoder with Self-Attention (Advanced Model)**
- **LSTM (Baseline Model)**

This project includes **positional encoding**, hyperparameter tuning, and evaluation using standard regression metrics.

---

## 📊 Dataset Description
- **Dataset Type:** Synthetic multivariate time series
- **Number of Features:** 3 variables (s1, s2, s3)
- **Dataset Size:** 6500+ observations
- **Preprocessing:** MinMax Normalization using `MinMaxScaler`

---

## ⚙️ Forecasting Setup
- **Input Sequence Length:** 48 time steps
- **Output Forecast Horizon:** 12 time steps (multi-step forecasting)

---

## 🧠 Models Implemented

### ✅ Transformer Forecast Model
- Uses **Transformer Encoder Layer**
- Includes **Multi-Head Self-Attention**
- Includes **Sinusoidal Positional Encoding**
- Learns long-term dependencies in time-series data

### ✅ LSTM Baseline Model
- Standard recurrent model for sequential forecasting
- Used for baseline comparison

---

## 📈 Evaluation Metrics
The models are evaluated using regression-based forecasting metrics:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**

Lower values indicate better forecasting performance.

---

## 🔍 Hyperparameter Tuning
The Transformer model was trained with multiple configurations:

- Epochs: 10 and 20
- Learning Rate: 0.001

The best model was selected based on the lowest RMSE.

---

## 🏁 Final Results (Test Set)

| Model        | MAE     | RMSE    |
|-------------|---------|---------|
| Transformer | 0.0335  | 0.0443  |
| LSTM        | 0.0273  | 0.0422  |

📌 Observation: In this dataset, the LSTM baseline slightly outperformed the Transformer model.

---

## 🛠 Tools & Libraries Used
- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---


## 📌 Conclusion
This project demonstrates the use of **attention-based Transformers** for multivariate time-series forecasting and provides a comparison with a traditional LSTM baseline. Results show that model performance depends on dataset characteristics, and baseline comparison is essential for fair evaluation.
