# Titanic Survival Prediction: Deep Neural Network from Scratch in NumPy

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?logo=Jupyter&logoColor=white)](https://jupyter.org/)

A zero-framework, first-principles implementation of a multi-layer deep neural network (Multi-Layer Perceptron) built entirely in **Python & NumPy** for binary survival classification on the Titanic dataset.

---

## Table of Contents
- [Overview](#overview)
- [Neural Network Architecture](#neural-network-architecture)
- [Mathematical Framework](#mathematical-framework)
  - [Weight Initialization (He & Xavier/Glorot)](#weight-initialization-he--xavierglorot)
  - [Forward Propagation](#forward-propagation)
  - [Loss Function (Binary Cross-Entropy)](#loss-function-binary-cross-entropy)
  - [Vectorized Backpropagation & Gradients](#vectorized-backpropagation--gradients)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Training Dynamics & Results](#training-dynamics--results)
- [License](#license)
- [Author](#author)

---

## Overview

Modern deep learning frameworks often abstract away computational graphs, automatic differentiation, and tensor calculus. 

This project implements an end-to-end neural network purely using linear algebra in NumPy to demonstrate:
1. **Analytical Gradient Derivation**: Full manual backpropagation using the multivariate chain rule.
2. **Numerical Stability**: Preventing floating-point underflow/overflow during log-loss computation and Sigmoid activation.
3. **Variance-Preserving Initialization**: Utilizing He uniform initialization for ReLU hidden layers and Xavier/Glorot uniform initialization for the Sigmoid output layer.
4. **Vectorized Matrix Computation**: Operating on batch matrices without iterative Python loops.

---

## Neural Network Architecture

A fully connected 3-layer feedforward network (Multi-Layer Perceptron):

```
Input Features (X)          [m × 8]
        │
   [W1, b1]  ──  He Uniform Init  ──  ReLU
        ▼
 Hidden Layer 1 (a1)        [m × 3]
        │
   [W2, b2]  ──  He Uniform Init  ──  ReLU
        ▼
 Hidden Layer 2 (a2)        [m × 3]
        │
   [W3, b3]  ──  Xavier Uniform Init  ──  Sigmoid
        ▼
Output Probability (ŷ)      [m × 1]  ──>  Binary Survival Prediction (0 or 1)
```

- **Input Dimension ($D_{\text{in}} = 8$)**: Preprocessed tabular passenger features.
- **Hidden Layer 1 ($H_1 = 3$)**: Rectified Linear Unit ($\text{ReLU}$) activation.
- **Hidden Layer 2 ($H_2 = 3$)**: Rectified Linear Unit ($\text{ReLU}$) activation.
- **Output Layer ($D_{\text{out}} = 1$)**: Sigmoid activation producing survival probability $\hat{y} \in (0, 1)$.

---

## Mathematical Framework

### Weight Initialization (He & Xavier/Glorot)

To ensure gradients neither vanish nor explode across layers, weights are initialized based on fan-in and fan-out dimensions:

For hidden layers with **ReLU** activation (Layers 1 and 2):

$$
W \sim \mathcal{U}\left(-\sqrt{\frac{6}{\text{fan}_{\text{in}}}}, \; \sqrt{\frac{6}{\text{fan}_{\text{in}}}}\right)
$$

For the output layer with **Sigmoid** activation (Layer 3):

$$
W_3 \sim \mathcal{U}\left(-\sqrt{\frac{6}{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}}, \; \sqrt{\frac{6}{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}}\right)
$$

Biases are initialized to zeros:

$$
b_1 = \mathbf{0}_{(1 \times 3)}, \quad b_2 = \mathbf{0}_{(1 \times 3)}, \quad b_3 = \mathbf{0}_{(1 \times 1)}
$$

---

### Forward Propagation

For a batch of $m$ passenger observations represented by matrix $X \in \mathbb{R}^{m \times 8}$:

**Layer 1 (Input to Hidden 1):**

$$
Z_1 = X W_1 + b_1
$$

$$
a_1 = \max(0, Z_1)
$$

**Layer 2 (Hidden 1 to Hidden 2):**

$$
Z_2 = a_1 W_2 + b_2
$$

$$
a_2 = \max(0, Z_2)
$$

**Layer 3 (Hidden 2 to Output):**

$$
Z_3 = a_2 W_3 + b_3
$$

$$
\hat{y} = \sigma(Z_3) = \frac{1}{1 + e^{-Z_3}}
$$

---

### Loss Function (Binary Cross-Entropy)

To ensure numerical stability when evaluating logarithms, predicted probabilities are clamped to $\hat{y} \in [\epsilon, 1 - \epsilon]$ where $\epsilon = 10^{-15}$. The binary cross-entropy loss $\mathcal{L}$ over $m$ samples is:

$$
\mathcal{L} = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]
$$

---

### Vectorized Backpropagation & Gradients

Using matrix differential calculus and the chain rule:

**Output Layer Gradients:**

$$
\frac{\partial \mathcal{L}}{\partial Z_3} = \frac{1}{m} (\hat{y} - y)
$$

$$
\frac{\partial \mathcal{L}}{\partial W_3} = a_2^T \frac{\partial \mathcal{L}}{\partial Z_3}
$$

$$
\frac{\partial \mathcal{L}}{\partial b_3} = \sum_{i=1}^m \frac{\partial \mathcal{L}}{\partial Z_3^{(i)}}
$$

**Hidden Layer 2 Gradients:**

$$
\frac{\partial \mathcal{L}}{\partial a_2} = \frac{\partial \mathcal{L}}{\partial Z_3} W_3^T
$$

$$
\frac{\partial \mathcal{L}}{\partial Z_2} = \frac{\partial \mathcal{L}}{\partial a_2} \odot \mathbb{I}(Z_2 > 0)
$$

$$
\frac{\partial \mathcal{L}}{\partial W_2} = a_1^T \frac{\partial \mathcal{L}}{\partial Z_2}
$$

$$
\frac{\partial \mathcal{L}}{\partial b_2} = \sum_{i=1}^m \frac{\partial \mathcal{L}}{\partial Z_2^{(i)}}
$$

**Hidden Layer 1 Gradients:**

$$
\frac{\partial \mathcal{L}}{\partial a_1} = \frac{\partial \mathcal{L}}{\partial Z_2} W_2^T
$$

$$
\frac{\partial \mathcal{L}}{\partial Z_1} = \frac{\partial \mathcal{L}}{\partial a_1} \odot \mathbb{I}(Z_1 > 0)
$$

$$
\frac{\partial \mathcal{L}}{\partial W_1} = X^T \frac{\partial \mathcal{L}}{\partial Z_1}
$$

$$
\frac{\partial \mathcal{L}}{\partial b_1} = \sum_{i=1}^m \frac{\partial \mathcal{L}}{\partial Z_1^{(i)}}
$$

**Parameter Update (Gradient Descent):**

$$
\theta \leftarrow \theta - \alpha \frac{\partial \mathcal{L}}{\partial \theta}
$$

with learning rate $\alpha = 0.05$.

---

## Data Preprocessing Pipeline

- **Feature Pruning**: Removes non-predictive metadata (`PassengerId`, `Name`, `Ticket`, `Cabin`).
- **Median Imputation**: Replaces missing values in `Age` and `Fare` with training-set medians.
- **Categorical Transformation**:
  - `Sex`: Mapped to binary integers ($\text{female} \to 1, \text{male} \to 0$).
  - `Embarked`: One-hot encoded across ports `C`, `S`, and `Q`.
- **Feature Standardization**:

$$
X_{\text{norm}} = \frac{X - \mu}{\sigma}
$$

where mean $\mu$ and standard deviation $\sigma$ are calculated on the training split to prevent data leakage.

---

## Project Structure

```
titanic-ml-from-scratch/
├── TitanicML.ipynb      # Complete Jupyter Notebook (data cleaning, neural net, training)
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── README.md            # Documentation and mathematical formulation
```

---

## Installation & Setup

```bash
# Clone repository
git clone https://github.com/MDiazEsperon/titanic-ml-from-scratch.git
cd titanic-ml-from-scratch

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

---

## Usage Guide

Launch the Jupyter Notebook to explore data preprocessing, model implementation, and training:

```bash
jupyter notebook TitanicML.ipynb
```

The notebook contains the complete pipeline:
1. Data loading and feature engineering with Pandas.
2. Parameter initialization.
3. 8,000-epoch training loop with forward pass, loss tracking, and backpropagation.
4. Validation loss monitoring.
5. Test dataset prediction and submission generation.

---

## Training Dynamics & Results

During 8,000 iterations of gradient descent:
- **Baseline Cross-Entropy Loss**: $\sim 0.693$ (untrained random initialization)
- **Converged Training Loss**: $\sim 0.408$
- **Validation Loss**: $\sim 0.390$
- **Classification Performance**: Achieves stable convergence without exploding gradients or vanishing activations.

---

## License

This project is open-source under the [MIT License](LICENSE).

---

## Author

**Miguel Díaz Esperón**  
- GitHub: [@MDiazEsperon](https://github.com/MDiazEsperon)  
- Repository: [titanic-ml-from-scratch](https://github.com/MDiazEsperon/titanic-ml-from-scratch)
