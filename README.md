# Predictive Maintenance: Machine Failure Classification

**Binary Classification | XGBoost | PyTorch | Optuna | Scikit-learn**

## 1. Project Overview

Unplanned equipment failures can significantly impact manufacturing productivity, increase maintenance costs, and disrupt production schedules.

This project explores the use of Machine Learning to identify potential machine failures based on operational measurements from the **AI4I 2020 Predictive Maintenance Dataset**.

The main objective is to develop and compare different binary classification models, evaluate their ability to identify failures under significant class imbalance, and investigate whether hyperparameter optimization can improve predictive performance.

Three models are developed and evaluated:

- Baseline XGBoost Classifier / Sci-kit Learn
- Fine-Tuned XGBoost Classifier / Sci-kit Learn
- Fine-Tuned Multilayer Perceptron (MLP) / PyTorch

The fine-tuning is done using `GridSearchCV` and `RandomizedSearchCV` within the Sci-kit Learn framework, and `Optuna` for the MLP.

## 2. Dataset

The project uses the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance), containing 10,000 synthetic observations representing industrial machine operating conditions.

The dataset includes the following operational variables:

| Feature | Description |
|---|---|
| Type | Machine quality variant |
| Air temperature | Ambient air temperature (K) |
| Process temperature | Process temperature (K) |
| Rotational speed | Rotational speed (rpm) |
| Torque | Torque (Nm) |
| Tool wear | Accumulated tool wear (min) |

The target variable is **Machine failure**, indicating whether a machine failure occurred.

Five additional failure-mode indicators are present in the original dataset. These are excluded from the model inputs to prevent target leakage.

The dataset is highly imbalanced, making precision and recall particularly important evaluation metrics.

## 3. Methodology

The project follows a structured Machine Learning workflow.

### Exploratory Data Analysis

- Feature distribution analysis
- Outlier inspection
- Identification of failure modes
- Examination of class imbalance

### Data Preparation and Feature Engineering

The dataset is divided into three subsets using stratified sampling:

- Training: 70%
- Validation: 15%
- Testing: 15%

Feature engineering includes:

- Mechanical power derived from torque and rotational speed.
- Temperature difference between process and ambient temperatures.
- Tool wear–torque interaction.
- Logarithmic transformation of rotational speed.

A Scikit-learn preprocessing pipeline handles imputation, feature scaling, and categorical encoding. The pipeline is fitted exclusively on the training set.

### XGBoost Classification

An initial XGBoost classifier establishes a baseline.

Hyperparameter optimization is subsequently performed using `RandomizedSearchCV`, exploring parameters such as:

- Learning rate
- Number of estimators
- Maximum tree depth
- Minimum child weight
- Subsampling
- Column sampling
- Regularization parameters
- Positive-class weighting

Five-fold cross-validation and Average Precision are used during the search.

### Neural Network Implementation

A feed-forward Multilayer Perceptron is implemented using PyTorch.

The network uses:
- ReLU activation functions
- Dropout regularization
- AdamW optimizer
- Binary cross-entropy with logits
- Positive-class weighting

GPU acceleration is used when CUDA is available.

Optuna is then employed to optimize the MLP architecture and training hyperparameters, including hidden-layer dimensions, dropout rates, learning rate, weight decay, and positive-class weighting.

### Classification Threshold Selection

Rather than relying exclusively on the default 0.5 classification threshold, multiple thresholds are evaluated using the validation set.

This allows the precision-recall trade-off to be examined and a threshold to be selected before final test-set evaluation.

## 4. Model Evaluation

The final models are evaluated on the held-out test set, which contains 1,500 observations, including 51 failures. 

### Metrics

Since the dataset has a big class imbalance (96.61% negatives and 3.39% positives), even a terrible classifier that outputs zeros all the time would get an accuracy score of 96.6%, thus this metric is not used. The metrics that the project focuses on are:

- F1: Ensure that the recall and precision metrics are well balanced, identifying failures and keeping false alarms under control
- Recall: Looking for a high recall will be indicative of a capable model that identifies a good percentage of failures.
- Average Precision: Summarizes the precision-recall trade-off across all classification thresholds into a single threshold-independent score

### Test Set Results

| Model | Threshold | Precision | Recall | F1 | Average Precision |
|---|---:|---:|---:|---:|---:|
| XGBoost Baseline | 0.65 | 0.89 | 0.82 | 0.86 | 0.87 |
| Fine-Tuned XGBoost | 0.70 | 0.88 | 0.86 | **0.87** | **0.90** |
| Fine-Tuned MLP | 0.25 | 0.71 | 0.69 | 0.70 | 0.77 |

### Key Findings

The fine-tuned XGBoost classifier achieved the strongest overall performance:

- **F1 Score:** 0.87
- **Precision:** 0.88
- **Recall:** 0.86
- **Average Precision:** 0.90

Compared with the baseline XGBoost model, hyperparameter optimization improved both recall and F1 score.

The MLP achieved an F1 score of 0.70, falling short of the tree-based models on this dataset.

These results suggest that XGBoost is better suited to this particular classification task among the evaluated approaches.

## 5. Limitations

Several limitations should be considered when interpreting these results:

- **Synthetic dataset:** The dataset is artificially generated and does not fully represent the complexity of real industrial environments.
- **Failure mechanisms:** Some failures are generated through predefined conditions, while random failures cannot necessarily be inferred from the available operational measurements.
- **Class imbalance:** The relatively small number of failure observations makes model evaluation particularly sensitive to false positives and false negatives.
- **Generalization:** Performance on this dataset does not guarantee equivalent results on real manufacturing equipment.

Further validation using real industrial data, different machines, and varying operating conditions would be necessary before considering production deployment.

## 6. Technologies and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- PyTorch
- TorchMetrics
- Optuna

## 7. Repository Structure

```text
AI4I2020-Predictive-Maintenance/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── notebooks/
│   └── AI4I2020_Predictive_Maintenance.ipynb
│
├── Datasets/
│   └── ai4i2020.csv
│
└── images/
    ├── confusion_matrix.png
    ├── model_comparison.png
    └── feature_distributions.png
```

## 8. Installation and Usage

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI4I2020-Predictive-Maintenance.git

cd AI4I2020-Predictive-Maintenance
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the AI4I 2020 dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance).

Place the CSV file inside the `Datasets/` directory using the filename:

```text
ai4i2020.csv
```

Open the notebook using Jupyter:

```bash
jupyter notebook
```

Run the notebook cells sequentially.

---

**Author:** Axel Vazquez 

**Domain:** Industrial Predictive Maintenance
