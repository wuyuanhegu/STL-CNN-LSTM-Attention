# STL-CNN-LSTM-Attention
The code is built on the PyTorch framework, and its core function is to perform monthly runoff prediction in the upper reaches of the Heihe River using a “fusion of STL time-series decomposition and deep learning models.” The code must be run with a dataset that has already undergone STL decomposition, and it automatically calculates metrics such as MAE, RMSE, R², NSE, and MAPE, ultimately outputting comparison charts and model weights. The specific usage instructions are as follows:
<img width="1200" height="609" alt="图四最终" src="https://github.com/user-attachments/assets/05d78836-a58d-4162-b995-244c49862405" />
1.Purpose of the Code
This code is used to implement the STL-CNN-LSTM-Attention hybrid prediction model proposed in the paper. Compared with using only a single CNN or LSTM model, this code loads a dataset containing STL decomposition features (trend component, seasonal component, and residual component) and multi-source environmental factors (SWVL1–SWVL4, SD), and constructs sequence feature engineering (including lagged terms and sliding-window statistics). Internally, the model sequentially uses a CNN to extract local features, a two-layer LSTM to learn long-term dependencies, and an attention mechanism to perform adaptive weighting over time steps, thereby achieving high-accuracy prediction of monthly runoff in the upper reaches of the Heihe River and systematic comparative evaluation of multiple models.
2.Runtime Environment and Data Preparation
Environment dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn, and torch (PyTorch) need to be installed.
Data format: The code reads from the default path F:\shuju\ql.xlsx. The input data table must contain the following fields: date, Trend, Seasonal, Residual, as well as swvl1–swvl4 and sd.
3.Parameter Configuration and Execution Workflow
Before use, the global parameters can be modified at the top of the code according to requirements. The code execution workflow is divided into four stages:
Data preprocessing stage: Lag features (Lag1–Lag3), differencing features, and rolling statistical features are constructed. After removing NaN values, the data are split by year into a training set (≤2000), a validation set (2001–2008), and a test set (≥2009), and the features are then standardized using Z-score normalization.
Model training stage: The Adam optimizer and MSE loss function are used, combined with ReduceLROnPlateau for dynamic learning rate adjustment and an early stopping mechanism (Patience=30). During training, the model weights with the best validation set performance are automatically saved.
Evaluation and figure generation stage: Evaluation metrics for each dataset are automatically calculated and saved; plotting functions are called to generate multiple figures, including the test set “actual vs. predicted values comparison plot,” “scatter calibration plot,” “residual distribution plot,” and “training loss curve.”
4.Results Output Description
After the code finishes running, all results will be automatically saved under the path specified by OUT_DIR:
\best_model: saves the optimal model weights cnn_lstm_attention_best.pt.
\tables: saves metrics.csv, predictions.csv, and split_summary.csv.
\charts: outputs various charts according to a unified academic color scheme, which can be directly used for paper figures or report presentations.
