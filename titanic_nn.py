"""
Titanic Survival Prediction: Deep Neural Network from Scratch in NumPy
Author: Miguel Díaz Esperón
License: MIT

Implements a 3-layer Multi-Layer Perceptron (MLP) from first principles using only
NumPy: custom forward pass, analytical backpropagation, He/Xavier parameter
initialization, numerical stability clipping, and binary classification.
"""

import argparse
import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


class TitanicNeuralNetwork:
    """
    Fully-connected Neural Network (Multi-Layer Perceptron) built from scratch.

    Architecture:
        - Input Layer: ft features
        - Hidden Layer 1: 3 neurons, ReLU activation, He uniform initialization
        - Hidden Layer 2: 3 neurons, ReLU activation, He uniform initialization
        - Output Layer: 1 neuron, Sigmoid activation, Xavier/Glorot uniform initialization
    """

    def __init__(self, input_dim: int, hidden_dim1: int = 3, hidden_dim2: int = 3, learning_rate: float = 0.05) -> None:
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.lr = learning_rate

        # He initialization for ReLU layers: bound = sqrt(6 / fan_in)
        h1 = np.sqrt(6.0 / input_dim)
        self.W1 = np.random.uniform(-h1, h1, (input_dim, hidden_dim1))
        self.b1 = np.zeros((1, hidden_dim1))

        h2 = np.sqrt(6.0 / hidden_dim1)
        self.W2 = np.random.uniform(-h2, h2, (hidden_dim1, hidden_dim2))
        self.b2 = np.zeros((1, hidden_dim2))

        # Xavier/Glorot initialization for Sigmoid output layer: bound = sqrt(6 / (fan_in + fan_out))
        x1 = np.sqrt(6.0 / (hidden_dim2 + 1))
        self.W3 = np.random.uniform(-x1, x1, (hidden_dim2, 1))
        self.b3 = np.zeros((1, 1))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Executes forward propagation pass."""
        Z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0.0, Z1)  # ReLU

        Z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0.0, Z2)  # ReLU

        Z3 = a2 @ self.W3 + self.b3
        opred = 1.0 / (1.0 + np.exp(-Z3))  # Sigmoid

        cache = {"X": X, "Z1": Z1, "a1": a1, "Z2": Z2, "a2": a2, "Z3": Z3, "opred": opred}
        return opred, cache

    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes Binary Cross-Entropy Loss with epsilon clipping for numerical stability."""
        clipped_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        loss = -np.mean(y_true * np.log(clipped_pred) + (1.0 - y_true) * np.log(1.0 - clipped_pred))
        return float(loss)

    def backward(self, cache: Dict[str, np.ndarray], y_true: np.ndarray) -> None:
        """Computes analytical gradients and applies gradient descent parameter updates."""
        m = len(cache["X"])
        opred = cache["opred"]
        a2 = cache["a2"]
        Z2 = cache["Z2"]
        a1 = cache["a1"]
        Z1 = cache["Z1"]
        X = cache["X"]

        # Output layer gradient (dL/dZ3)
        d_Z3 = (opred - y_true) / m
        d_W3 = a2.T @ d_Z3
        d_b3 = np.sum(d_Z3, axis=0, keepdims=True)

        # Hidden layer 2 gradient
        d_a2 = d_Z3 @ self.W3.T
        d_Z2 = d_a2 * (Z2 > 0)
        d_W2 = a1.T @ d_Z2
        d_b2 = np.sum(d_Z2, axis=0, keepdims=True)

        # Hidden layer 1 gradient
        d_a1 = d_Z2 @ self.W2.T
        d_Z1 = d_a1 * (Z1 > 0)
        d_W1 = X.T @ d_Z1
        d_b1 = np.sum(d_Z1, axis=0, keepdims=True)

        # Gradient descent parameter updates
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W3 -= self.lr * d_W3
        self.b3 -= self.lr * d_b3

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 8000,
        eval_every: int = 500,
    ) -> None:
        """Trains the neural network using mini-batch/full-batch gradient descent."""
        print(f"Beginning training for {epochs} epochs (learning rate = {self.lr})...")
        for epoch in range(epochs):
            y_pred, cache = self.forward(X_train)
            self.backward(cache, y_train)

            if epoch % eval_every == 0 or epoch == epochs - 1:
                train_loss = self.compute_loss(y_train, y_pred)
                train_acc = np.mean((y_pred >= 0.5) == y_train) * 100.0
                val_info = ""
                if X_val is not None and y_val is not None:
                    val_pred, _ = self.forward(X_val)
                    val_loss = self.compute_loss(y_val, val_pred)
                    val_acc = np.mean((val_pred >= 0.5) == y_val) * 100.0
                    val_info = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%"
                print(f"Epoch {epoch:5d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%{val_info}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Outputs binary class labels (0 or 1)."""
        probabilities, _ = self.forward(X)
        return (probabilities >= 0.5).astype(int)


def preprocess_data(df: pd.DataFrame, train_stats: Optional[Tuple[pd.Series, pd.Series]] = None) -> Tuple[np.ndarray, Tuple[pd.Series, pd.Series]]:
    """Cleans, encodes categorical features, and standardizes numerical columns."""
    df_clean = df.copy()

    # Drop non-predictive metadata
    cols_to_drop = [c for c in ["PassengerId", "Name", "Ticket", "Cabin", "Survived"] if c in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)

    # Impute missing values with medians
    if "Age" in df_clean.columns:
        df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())
    if "Fare" in df_clean.columns:
        df_clean["Fare"] = df_clean["Fare"].fillna(df_clean["Fare"].median())

    # Categorical encoding
    if "Sex" in df_clean.columns:
        df_clean["Sex"] = df_clean["Sex"].map({"female": 1, "male": 0}).fillna(0).astype(int)

    if "Embarked" in df_clean.columns:
        df_clean["Embarked"] = df_clean["Embarked"].fillna("S")
        df_clean["Embarked_C"] = (df_clean["Embarked"] == "C").astype(int)
        df_clean["Embarked_S"] = (df_clean["Embarked"] == "S").astype(int)
        df_clean["Embarked_Q"] = (df_clean["Embarked"] == "Q").astype(int)
        df_clean = df_clean.drop(columns=["Embarked"])

    # Feature standardization: (X - mu) / sigma
    if train_stats is None:
        mean = df_clean.mean()
        std = df_clean.std().replace(0, 1.0)
    else:
        mean, std = train_stats

    df_normalized = (df_clean - mean) / std
    return df_normalized.to_numpy(), (mean, std)


def generate_synthetic_data(num_samples: int = 891) -> pd.DataFrame:
    """Generates synthetic tabular Titanic-like dataset for demonstration when raw CSV is absent."""
    np.random.seed(42)
    pclass = np.random.choice([1, 2, 3], size=num_samples, p=[0.24, 0.21, 0.55])
    sex = np.random.choice(["female", "male"], size=num_samples, p=[0.35, 0.65])
    age = np.clip(np.random.normal(29.7, 14.5, size=num_samples), 1, 80)
    sibsp = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.68, 0.23, 0.05, 0.02, 0.02])
    parch = np.random.choice([0, 1, 2], size=num_samples, p=[0.76, 0.13, 0.11])
    fare = np.clip(np.random.exponential(32.0, size=num_samples), 4, 512)
    embarked = np.random.choice(["S", "C", "Q"], size=num_samples, p=[0.72, 0.19, 0.09])

    # Realistic survival probability based on historical correlations
    logit = 1.2 * (sex == "female") - 0.7 * (pclass - 1) - 0.02 * age + 0.005 * fare - 0.3
    prob = 1.0 / (1.0 + np.exp(-logit))
    survived = (np.random.rand(num_samples) < prob).astype(int)

    return pd.DataFrame({
        "PassengerId": np.arange(1, num_samples + 1),
        "Survived": survived,
        "Pclass": pclass,
        "Name": [f"Passenger {i}" for i in range(num_samples)],
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": ["LINE"] * num_samples,
        "Fare": fare,
        "Cabin": [np.nan] * num_samples,
        "Embarked": embarked,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Titanic Deep Neural Network from Scratch in NumPy")
    parser.add_argument("--train-path", type=str, default="train_titanicml.csv", help="Path to training CSV")
    parser.add_argument("--test-path", type=str, default="test_titanicml.csv", help="Path to testing CSV")
    parser.add_argument("--epochs", type=int, default=8000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    args = parser.parse_args()

    # Load data
    if os.path.exists(args.train_path):
        print(f"Loading training data from {args.train_path}...")
        raw_df = pd.read_csv(args.train_path)
    else:
        print(f"'{args.train_path}' not found on disk. Generating synthetic data for demonstration...")
        raw_df = generate_synthetic_data()

    y = raw_df["Survived"].to_numpy().reshape(-1, 1)

    # Train / Validation split (85% train, 15% validation)
    split_idx = int(0.85 * len(raw_df))
    df_train_raw = raw_df.iloc[:split_idx]
    y_train = y[:split_idx]

    df_val_raw = raw_df.iloc[split_idx:]
    y_val = y[split_idx:]

    X_train, stats = preprocess_data(df_train_raw)
    X_val, _ = preprocess_data(df_val_raw, train_stats=stats)

    feature_dim = X_train.shape[1]
    print(f"Dataset preprocessed: {len(X_train)} train samples, {len(X_val)} validation samples, {feature_dim} features.")

    model = TitanicNeuralNetwork(input_dim=feature_dim, learning_rate=args.lr)
    model.fit(X_train, y_train, X_val, y_val, epochs=args.epochs, eval_every=500)

    # Predict test data if present
    if os.path.exists(args.test_path):
        print(f"\nEvaluating test dataset from {args.test_path}...")
        test_df = pd.read_csv(args.test_path)
        X_test, _ = preprocess_data(test_df, train_stats=stats)
        predictions = model.predict(X_test)
        submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": predictions.flatten()})
        submission.to_csv("submission.csv", index=False)
        print("Generated Kaggle competition submission saved to 'submission.csv'.")
    else:
        print("\nTraining completed successfully! (No test file found to generate submission.csv)")


if __name__ == "__main__":
    main()
