from __future__ import annotations
import math
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

RAW_FILE = Path(r"F:\shuju\ql.xlsx")
TARGET_COL = "Runoff"
LOOKBACK = 12
HORIZON = 1
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
DROPOUT_RATE = 0.2
RANDOM_SEED = 42
WEIGHT_DECAY = 3e-4

OUT_DIR = Path(r"F:\jieguo\STL-CNN-LSTM-Attention\ql")
TABLE_DIR = OUT_DIR / "tables"
CHART_DIR = OUT_DIR / "charts"
MODEL_DIR = OUT_DIR / "best_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 颜色主题
TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLOR_FAMILIES = {
    "blue": {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "gold": {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "orange": {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
    "pink": {"xlight": "#FCDAD6", "light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"},
}
@dataclass
class SequenceSplit:
    X: np.ndarray
    y: np.ndarray
    timestamps: pd.Series
@dataclass
class SplitBundle:
    train: SequenceSplit
    val: SequenceSplit
    test: SequenceSplit

@dataclass
class ModelRunResult:
    status: str
    model_name: str
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    training_history: pd.DataFrame
    blocker_message: str = ""

def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Segoe UI", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
        },
    )


def add_chart_header(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    title = textwrap.fill(str(title), width=72, break_long_words=False)
    subtitle = textwrap.fill(str(subtitle), width=108, break_long_words=False)
    ax.set_title("")
    fig.subplots_adjust(top=0.82)
    left = ax.get_position().x0
    fig.text(left, 0.98, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.93, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    sns.despine(ax=ax)

def load_raw_data(path: Path = RAW_FILE) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df

def prepare_model_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = raw.copy()
    required_cols = [
        "date", "Runoff", "Trend", "Seasonal", "Residual",
         "swvl1", "swvl2", "swvl3", "swvl4", "sd"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少字段: {col}")

    df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["month"] = df["timestamp"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    base_features = [
        "Trend", "Seasonal", "Residual",
        "swvl1", "swvl2", "swvl3", "swvl4", "sd"
    ]

    df["Runoff_lag1"] = df[TARGET_COL].shift(1)
    df["Runoff_lag2"] = df[TARGET_COL].shift(2)
    df["Runoff_lag3"] = df[TARGET_COL].shift(3)

    df["Runoff_diff1"] = df["Runoff_lag1"] - df["Runoff_lag2"]
    df["Runoff_diff2"] = df["Runoff_lag2"] - df["Runoff_lag3"]

    df["Runoff_roll_mean3"] = df[TARGET_COL].shift(1).rolling(3).mean()
    df["Runoff_roll_std3"] = df[TARGET_COL].shift(1).rolling(3).std()

    df["Runoff_ratio1"] = df["Runoff_lag1"] / (df["Runoff_lag2"] + 1e-6)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    for col in base_features + [TARGET_COL, "month_sin", "month_cos"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET_COL] = df[TARGET_COL].astype(float)

    feature_cols = base_features + [
        "Runoff_lag1", "Runoff_lag2", "Runoff_lag3",
        "Runoff_diff1", "Runoff_diff2",
        "Runoff_roll_mean3", "Runoff_roll_std3",
        "Runoff_ratio1",
        "month_sin", "month_cos"
    ]
    df[feature_cols] = df[feature_cols].astype(float)
    return df, feature_cols

def create_sequences(
        frame: pd.DataFrame,
        feature_cols: list[str],
        lookback: int = LOOKBACK,
        horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:

    if lookback < 1 or horizon < 1:
        raise ValueError("lookback 和 horizon 必须 >= 1")
    features = frame[feature_cols].to_numpy(dtype=np.float32)
    target = frame[TARGET_COL].to_numpy(dtype=np.float32)
    timestamps = frame["timestamp"].reset_index(drop=True)

    X_list = []
    y_list = []
    ts_list = []
    last_start = len(frame) - lookback - horizon + 1
    for start in range(max(0, last_start)):
        target_idx = start + lookback + horizon - 1
        y_val = target[target_idx]
        if np.isnan(y_val):
            continue
        X_list.append(features[start:start + lookback])
        y_list.append(float(y_val))
        ts_list.append(timestamps.iloc[target_idx])
    if not X_list:
        raise ValueError("没有生成任何有效样本，请检查 lookback 和 horizon 是否过大。")
    return np.stack(X_list), np.asarray(y_list, dtype=np.float32), pd.Series(ts_list, name="timestamp")


def split_sequences_by_year(X, y, timestamps):
    years = pd.to_datetime(timestamps).dt.year
    train_mask = years <= 2000
    val_mask = (years >= 2001) & (years <= 2008)
    test_mask = years >= 2009
    return SplitBundle(
        train=SequenceSplit(X[train_mask], y[train_mask], timestamps[train_mask].reset_index(drop=True)),
        val=SequenceSplit(X[val_mask], y[val_mask], timestamps[val_mask].reset_index(drop=True)),
        test=SequenceSplit(X[test_mask], y[test_mask], timestamps[test_mask].reset_index(drop=True))
    )


def standardize_splits(splits: SplitBundle) -> tuple[SplitBundle, StandardScaler]:
    scaler = StandardScaler()
    train_flat = splits.train.X.reshape(-1, splits.train.X.shape[-1])
    scaler.fit(train_flat)

    def transform(split: SequenceSplit) -> SequenceSplit:
        shape = split.X.shape
        scaled = scaler.transform(split.X.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)
        return SequenceSplit(scaled, split.y.astype(np.float32), split.timestamps)

    return SplitBundle(transform(splits.train), transform(splits.val), transform(splits.test)), scaler

def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    y_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)

    mae = np.sum(np.abs(y_true - y_pred)) / n

    rmse = np.sqrt(np.sum((y_true - y_pred) ** 2) / n)

    cov = np.sum((y_true - y_mean) * (y_pred - y_pred_mean))
    std_true = np.sqrt(np.sum((y_true - y_mean) ** 2))
    std_pred = np.sqrt(np.sum((y_pred - y_pred_mean) ** 2))

    if std_true > 1e-9 and std_pred > 1e-9:
        r2 = (cov / (std_true * std_pred)) ** 2
    else:
        r2 = np.nan

    nse = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_mean) ** 2))

    non_zero = np.abs(y_true) > 1e-9
    if np.any(non_zero):
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
    else:
        mape = np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "NSE": float(nse),
        "MAPE_%": float(mape)
    }

def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("请安装 PyTorch（推荐 CPU 版本）") from exc
    return torch

def build_torch_model_class(torch):
    nn = torch.nn

    class TemporalAttention(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.score = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        def forward(self, lstm_output):
            raw_scores = self.score(lstm_output).squeeze(-1)
            weights = torch.softmax(raw_scores, dim=1)
            context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)
            return context, weights

    class CNNLSTMAttention(nn.Module):
        def __init__(self, input_size: int, conv_channels: int = 32, hidden_size: int = 96,
                     dropout: float = DROPOUT_RATE):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.lstm = nn.LSTM(
                conv_channels, hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=dropout
            )
            self.lstm_dropout = nn.Dropout(dropout)
            self.attention = TemporalAttention(hidden_size)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            conv_in = x.transpose(1, 2)
            conv_out = self.conv(conv_in).transpose(1, 2)
            lstm_out, _ = self.lstm(conv_out)
            lstm_out = self.lstm_dropout(lstm_out)
            context, weights = self.attention(lstm_out)
            pred = self.fc(context).squeeze(-1)
            return pred, weights

    return CNNLSTMAttention

def train_cnn_lstm_attention(
        splits: SplitBundle,
        input_size: int,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        show_progress: bool = False,
) -> ModelRunResult:
    torch = require_torch()
    nn = torch.nn
    DataLoader = torch.utils.data.DataLoader
    TensorDataset = torch.utils.data.TensorDataset

    set_seed()
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_mean = float(splits.train.y.mean())
    target_std = float(splits.train.y.std() if splits.train.y.std() > 1e-6 else 1.0)

    def scale_y(values: np.ndarray) -> np.ndarray:
        return ((values - target_mean) / target_std).astype(np.float32)

    def inverse_scale_y(scaled: np.ndarray) -> np.ndarray:
        return scaled * target_std + target_mean

    train_ds = TensorDataset(
        torch.tensor(splits.train.X),
        torch.tensor(scale_y(splits.train.y))
    )
    val_x = torch.tensor(splits.val.X, device=device)
    val_y = torch.tensor(scale_y(splits.val.y), device=device)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    ModelClass = build_torch_model_class(torch)
    model = ModelClass(input_size=input_size).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=True
    )

    best_val = float("inf")
    best_state = None
    patience = 30
    wait = 0
    history_rows = []

    if show_progress:
        print(f"[训练] 开始训练：epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_pred, _ = model(val_x)
            val_loss = float(loss_fn(val_pred, val_y).detach().cpu().item())

        scheduler.step(val_loss)

        train_loss = float(np.mean(batch_losses))
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            improved = True
        else:
            wait += 1
            improved = False

        if wait >= patience:
            if show_progress:
                print(f"[训练] Epoch {epoch:03d}/{epochs} 早停，最佳验证损失 {best_val:.6f}")
            break

        if show_progress:
            marker = "best" if improved else f"wait={wait}/{patience}"
            print(
                f"[训练] Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | {marker}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict(split: SequenceSplit, split_name: str) -> tuple[pd.DataFrame, dict[str, float]]:
        with torch.no_grad():
            pred_scaled, attn = model(torch.tensor(split.X, device=device))
            pred = inverse_scale_y(pred_scaled.detach().cpu().numpy())
            attn_np = attn.detach().cpu().numpy()
        metrics = compute_regression_metrics(split.y, pred)
        pred_df = pd.DataFrame({
            "split": split_name,
            "timestamp": split.timestamps,
            "actual": split.y,
            "predicted": pred,
            "residual": split.y - pred,
            "attention_last_step": attn_np[:, -1] if attn_np.shape[1] > 0 else np.nan,
            "attention_max": attn_np.max(axis=1),
        })
        return pred_df, metrics

    pred_frames = []
    metric_rows = []
    for split_name, split in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
        pred_df, metrics = predict(split, split_name)
        pred_frames.append(pred_df)
        metric_rows.append({"model": "CNN-LSTM-Attention", "split": split_name, **metrics})

    # 保存最佳模型
    model_path = MODEL_DIR / "cnn_lstm_attention_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "target_mean": target_mean,
        "target_std": target_std,
        "input_size": input_size,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
    }, model_path)
    if show_progress:
        print(f"[训练] 最佳模型已保存至 {model_path}")

    return ModelRunResult(
        status="executed",
        model_name="CNN-LSTM-Attention",
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(pred_frames, ignore_index=True),
        training_history=pd.DataFrame(history_rows),
    )

def table_split_summary(splits: SplitBundle) -> pd.DataFrame:
    rows = []
    total = len(splits.train.y) + len(splits.val.y) + len(splits.test.y)
    for name, split in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
        rows.append({
            "数据集": name,
            "样本数": len(split.y),
            "占比": len(split.y) / total,
            "开始时间": split.timestamps.min(),
            "结束时间": split.timestamps.max(),
            "目标均值": float(split.y.mean()),
            "目标中位数": float(np.median(split.y)),
            "目标最大值": float(split.y.max()),
        })
    return pd.DataFrame(rows)


def chart_split_counts(split_summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    colors = [COLOR_FAMILIES[name]["base"] for name in ["blue", "gold", "orange"]]
    edges = [COLOR_FAMILIES[name]["dark"] for name in ["blue", "gold", "orange"]]
    bars = ax.bar(split_summary["数据集"], split_summary["样本数"], color=colors, edgecolor=edges, linewidth=1.0)
    for bar, val in zip(bars, split_summary["样本数"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02 * max(split_summary["样本数"]), f"{int(val)}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("数据集")
    ax.set_ylabel("样本数")
    add_chart_header(fig, ax, "训练/验证/测试样本划分", "按时间顺序切分")
    path = CHART_DIR / "01_split_sample_counts.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_training_history(history: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    if history.empty:
        ax.axis("off")
        ax.text(0.05, 0.6, "训练历史为空", fontsize=14, color=TOKENS["ink"], transform=ax.transAxes)
    else:
        sns.lineplot(data=history, x="epoch", y="train_loss", color=COLOR_FAMILIES["blue"]["base"], marker="o",
                     label="train loss", ax=ax)
        sns.lineplot(data=history, x="epoch", y="val_loss", color=COLOR_FAMILIES["orange"]["base"], marker="o",
                     label="val loss", ax=ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_xlim(left=0)
        ax.margins(x=0, y=0)
        ax.legend(frameon=False)
    add_chart_header(fig, ax, "训练曲线", "训练集与验证集损失")
    path = CHART_DIR / "03_training_history.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path

def chart_actual_vs_pred(predictions: pd.DataFrame, filename: str = "04_actual_vs_pred.png") -> Path:
    test = predictions[predictions["split"].eq("test")].copy()
    fig, ax = plt.subplots(figsize=(14, 6))
    if test.empty:
        ax.axis("off")
        ax.text(0.05, 0.6, "无测试集预测", fontsize=14)
    else:
        test = test.sort_values("timestamp").reset_index(drop=True)
        x = np.arange(len(test))
        ax.plot(x, test["actual"].values, label="实际值", linewidth=1.2)
        ax.plot(x, test["predicted"].values, label="预测值", linewidth=1.2)

        ax.set_xlabel("时长/月")
        ax.set_ylabel(r"流量/（m³·s$^{-1}$）")
        n_points = len(test)
        upper = int(math.ceil(n_points / 12.0)) * 12
        if upper < 12:
            upper = 12
        ax.set_xlim(0, upper)
        ticks = np.arange(0, upper + 1, 12)
        xtick_labels = ['' if t == 0 else str(int(t)) for t in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(xtick_labels)
        y_max = max(test["actual"].max(), test["predicted"].max())
        if y_max > 0:
            raw_step = y_max / 8
            if raw_step <= 1:
                step = 1
            elif raw_step <= 2:
                step = 2
            elif raw_step <= 5:
                step = 5
            elif raw_step <= 10:
                step = 10
            elif raw_step <= 20:
                step = 20
            elif raw_step <= 50:
                step = 50
            else:
                step = 100
            y_ticks = np.arange(0, y_max + step, step)
            if y_ticks[-1] < y_max:
                y_ticks = np.append(y_ticks, y_ticks[-1] + step)
        else:
            y_ticks = [0]
        ax.set_yticks(y_ticks)
        ax.set_ylim(0, y_ticks[-1])

        ax.legend()
    add_chart_header(fig, ax, "测试集预测对比", "")
    path = CHART_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path

def chart_scatter(predictions: pd.DataFrame, filename: str = "05_scatter.png") -> Path:
    test = predictions[predictions["split"].eq("test")].copy()
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    if test.empty:
        ax.axis("off")
        ax.text(0.05, 0.6, "无数据", fontsize=14, color=TOKENS["ink"], transform=ax.transAxes)
    else:
        sample = test.sample(min(500, len(test)), random_state=RANDOM_SEED)
        sns.scatterplot(
            data=sample, x="actual", y="predicted",
            color=COLOR_FAMILIES["pink"]["base"],
            edgecolor=COLOR_FAMILIES["pink"]["dark"],
            linewidth=0.3, alpha=0.5, ax=ax
        )
        upper = 100
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_aspect("equal", adjustable="box")

        ticks = np.arange(0, upper + 20, 20)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        xtick_labels = ['' if t == 0 else str(int(t)) for t in ticks]
        ytick_labels = [str(int(t)) for t in ticks]
        ax.set_xticklabels(xtick_labels)
        ax.set_yticklabels(ytick_labels)
        ax.plot([0, upper], [0, upper], linestyle=":", color=TOKENS["ink"], linewidth=1.2)
        ax.set_xlabel("实际值")
        ax.set_ylabel("预测值")
    add_chart_header(fig, ax, "预测散点校准", "")
    path = CHART_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_residual_distribution(predictions: pd.DataFrame, filename: str = "06_residual_dist.png") -> Path:
    test = predictions[predictions["split"].eq("test")].copy()
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    if test.empty:
        ax.axis("off")
        ax.text(0.05, 0.6, "无数据", fontsize=14, color=TOKENS["ink"], transform=ax.transAxes)
    else:
        sns.histplot(test["residual"], bins=30, color=COLOR_FAMILIES["gold"]["base"],
                     edgecolor=COLOR_FAMILIES["gold"]["dark"], linewidth=0.7, ax=ax)
        ax.axvline(0, color=TOKENS["ink"], linestyle=":", linewidth=1)
        ax.set_xlabel("残差")
        ax.set_ylabel("频数")
    add_chart_header(fig, ax, "测试集残差分布", "残差围绕0分布")
    path = CHART_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path

def main():
    print("=" * 60)
    print("CNN-LSTM-Attention 模型训练")
    print("=" * 60)

    print("1. 加载数据...")
    raw = load_raw_data()
    print(f" 原始数据形状: {raw.shape}")

    print("2. 特征工程...")
    frame, feature_cols = prepare_model_frame(raw)
    print(f" 特征列数: {len(feature_cols)}")
    print(f" 数据时间范围: {frame['timestamp'].min()} 至 {frame['timestamp'].max()}")

    print(f"3. 生成序列 (lookback={LOOKBACK}, horizon={HORIZON})...")
    X, y, timestamps = create_sequences(frame, feature_cols, LOOKBACK, HORIZON)
    print(f" 生成样本数: {len(y)}")

    print("4. 划分训练/验证/测试...")
    splits = split_sequences_by_year(X, y, timestamps)
    print(f" 训练: {len(splits.train.y)}, 验证: {len(splits.val.y)}, 测试: {len(splits.test.y)}")

    print("5. 标准化特征...")
    splits, scaler = standardize_splits(splits)

    print("6. 训练模型...")
    result = train_cnn_lstm_attention(
        splits,
        input_size=len(feature_cols),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        show_progress=True,
    )

    print("\n7. 评估结果:")
    print(result.metrics.to_string(index=False))

    print("8. 生成报告图表...")
    use_chart_theme()

    split_summary = table_split_summary(splits)
    split_summary.to_csv(TABLE_DIR / "split_summary.csv", index=False, encoding="utf-8-sig")
    chart_split_counts(split_summary)

    result.training_history.to_csv(TABLE_DIR / "training_history.csv", index=False, encoding="utf-8-sig")
    chart_training_history(result.training_history)

    result.predictions.to_csv(TABLE_DIR / "predictions.csv", index=False, encoding="utf-8-sig")
    chart_actual_vs_pred(result.predictions)
    chart_scatter(result.predictions)
    chart_residual_distribution(result.predictions)

    result.metrics.to_csv(TABLE_DIR / "metrics.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("训练完成！所有结果保存在：", OUT_DIR.resolve())
    print("=" * 60)


if __name__ == "__main__":
    main()