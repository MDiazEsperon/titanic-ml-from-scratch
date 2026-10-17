# Titanic Survival Prediction: Deep Neural Network from Scratch in NumPy

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?logo=Jupyter&logoColor=white)](https://jupyter.org/)

A zero-framework, first-principles implementation of a multi-layer deep neural network (MLP) built entirely in **Python & NumPy** for binary survival classification on the Kaggle Titanic dataset.

---

## Table of Contents
- [Why Build From Scratch?](#why-build-from-scratch)
- [Neural Network Architecture](#neural-network-architecture)
- [Mathematical Derivations](#mathematical-derivations)
  - [He & Xavier/Glorot Initialization](#he--xavierglorot-initialization)
  - [Forward Propagation](#forward-propagation)
  - [Loss Formulation (Binary Cross-Entropy)](#loss-formulation-binary-cross-entropy)
  - [Analytical Backpropagation & Gradients](#analytical-backpropagation--gradients)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
  - [Running the Interactive Jupyter Notebook](#running-the-interactive-jupyter-notebook)
  - [Running the CLI Pipeline](#running-the-cli-pipeline)
- [Model Training & Results](#model-training--results)
- [License](#license)
- [Author](#author)

---

## Why Build From Scratch?

Modern machine learning libraries (PyTorch, TensorFlow, Scikit-Learn) abstract away the underlying computational graph, autograd engines, and linear algebra routines.

This project implements every component manually to demonstrate deep understanding of:
1. **Matrix calculus & analytical chain-rule gradient derivation**.
2. **Numerical stability guards** (preventing floating-point underflow/overflow in log-loss and sigmoid).
3. **Weight initialization theory** (variance preservation across non-linear activation functions).
4. **Vectorized tensor computation** without high-level framework abstractions.

---

## Neural Network Architecture

A fully-connected 3-layer feedforward architecture (Multi-Layer Perceptron):

```
Input Features (X)       [m × 8]
        │
   [W1, b1]  ──  He Uniform Init  ──  ReLU
        ▼
 Hidden Layer 1 (a1)     [m × 3]
        │
   [W2, b2]  ──  He Uniform Init  ──  ReLU
        ▼
 Hidden Layer 2 (a2)     [m × 3]
        │
   [W3, b3]  ──  Xavier Uniform Init  ──  Sigmoid
        ▼
Output Probability (ŷ)   [m × 1]  ──>  Binary Survival Class (0 or 1)
```

- **Input Dimension ($D_{\text{in}} = 8$)**: Preprocessed tabular passenger features.
- **Hidden Layer 1 ($H_1 = 3$)**: Rectified Linear Unit ($\text{ReLU}$) activation.
- **Hidden Layer 2 ($H_2 = 3$)**: Rectified Linear Unit ($\text{ReLU}$) activation.
- **Output Layer ($D_{\text{out}} = 1$)**: Sigmoid activation producing calibrated survival probabilities $\hat{y} \in (0, 1)$.

---

## Mathematical Derivations

### He & Xavier/Glorot Initialization

To prevent vanishing or exploding gradients during backpropagation:
- **He Initialization (Layers 1 & 2 - ReLU)**:
  $$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{\text{fan\_in}}}, \; \sqrt{\frac{6}{\text{fan\_in}}}\right)$$
- **Xavier / Glorot Initialization (Layer 3 - Sigmoid)**:
  $$W_3 \sim \mathcal{U}\left(-\sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}, \; \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}\right)$$

### Forward Propagation
For a batch of $m$ passenger records:

$$\mathbf{Z}_1 = \mathbf{X} \mathbf{W}_1 + \mathbf{b}_1, \quad \mathbf{a}_1 = \max(0, \mathbf{Z}_1)$$

$$\mathbf{Z}_2 = \mathbf{a}_1 \mathbf{W}_2 + \mathbf{b}_2, \quad \mathbf{a}_2 = \max(0, \mathbf{Z}_2)$$

$$\mathbf{Z}_3 = \mathbf{a}_2 \mathbf{W}_3 + \mathbf{b}_3, \quad \hat{\mathbf{y}} = \sigma(\mathbf{Z}_3) = \frac{1}{1 + e^{-\mathbf{Z}_3}}$$

### Loss Formulation (Binary Cross-Entropy)
With numerical stability clipping $\hat{y}_{\text{clipped}} = \text{clip}(\hat{y}, 10^{-15}, 1 - 10^{-15})$:

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

### Analytical Backpropagation & Gradients
Applying matrix differential calculus and the chain rule:

1. **Output Error**:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{Z}_3} = \frac{1}{m} (\hat{\mathbf{y}} - \mathbf{y})$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_3} = \mathbf{a}_2^T \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_3}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}_3} = \sum_{\text{rows}} \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_3}$$

2. **Hidden Layer 2**:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{a}_2} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_3} \mathbf{W}_3^T$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{Z}_2} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}_2} \odot \mathbb{I}(\mathbf{Z}_2 > 0)$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = \mathbf{a}_1^T \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_2}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}_2} = \sum_{\text{rows}} \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_2}$$

3. **Hidden Layer 1**:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{a}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_2} \mathbf{W}_2^T$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{Z}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}_1} \odot \mathbb{I}(\mathbf{Z}_1 > 0)$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \mathbf{X}^T \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_1}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \sum_{\text{rows}} \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_1}$$

4. **Parameter Updates (Gradient Descent)**:
   $$\theta \leftarrow \theta - \alpha \frac{\partial \mathcal{L}}{\partial \theta}, \quad \alpha = 0.05$$

---

## Data Preprocessing Pipeline

- **Feature Selection**: Drops non-generalizable textual identifiers (`PassengerId`, `Name`, `Ticket`, `Cabin`).
- **Missing Value Imputation**: Median strategy for `Age` and `Fare` to resist outlier skew.
- **Categorical Encoding**:
  - `Sex`: Binary transformation ($\text{female} \to 1, \text{male} \to 0$).
  - `Embarked`: One-hot encoding (`Embarked_C`, `Embarked_S`, `Embarked_Q`).
- **Feature Standardization**:
  $$\mathbf{X}_{\text{norm}} = \frac{\mathbf{X} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
  *(Parameters $\mu, \sigma$ are strictly fit on the training split and propagated to validation/test sets to eliminate data leakage).*

---

## Project Structure

```
titanic-ml-from-scratch/
├── TitanicML.ipynb      # Step-by-step interactive Jupyter Notebook
├── titanic_nn.py        # Modular production CLI training & evaluation script
├── requirements.txt     # Python dependencies (NumPy, Pandas, Matplotlib, Jupyter)
├── LICENSE              # MIT License
└── README.md            # Mathematical formulation & documentation
```

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/MEsperon-Code/titanic-ml-from-scratch.git
cd titanic-ml-from-scratch

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage Guide

### Running the Interactive Jupyter Notebook
Launch the notebook to inspect cell-by-cell execution, loss convergence, and derivations:
```bash
jupyter notebook TitanicML.ipynb
```

### Running the CLI Pipeline
You can run the model directly from the command line:
```bash
# Run training (uses train_titanicml.csv or generates synthetic validation data if CSV absent)
python titanic_nn.py --epochs 8000 --lr 0.05
```

Optional CLI parameters:
- `--train-path`: Custom path to training CSV (default: `train_titanicml.csv`).
- `--test-path`: Custom path to test CSV (default: `test_titanicml.csv`).
- `--epochs`: Total iterations for gradient descent (default: `8000`).
- `--lr`: Learning rate step size (default: `0.05`).

---

## Model Training & Results

During 8,000 epochs of training, the model achieves stable convergence:
- **Initial Loss**: $\sim 0.69$ (random guess baseline)
- **Converged Training Loss**: $\sim 0.40$
- **Validation Loss**: $\sim 0.39$
- **Classification Accuracy**: $\sim 81–83\%$ on held-out validation samples.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Miguel Díaz Esperón**  
- GitHub: [@MEsperon-Code](https://github.com/MEsperon-Code)
- Repository: [titanic-ml-from-scratch](https://github.com/MEsperon-Code/titanic-ml-from-scratch)
