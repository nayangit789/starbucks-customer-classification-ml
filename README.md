# ☕ Starbucks Customer Classification ML

A machine learning project that classifies Starbucks customers based on their ordering patterns to identify high-spending customers.

## 📋 Project Overview

This project analyzes Starbucks customer ordering patterns and builds a **Random Forest classification model** to predict whether a customer is a high-spender or not. The classification is based on the customer's total spending relative to the median spending across all customers.

### Key Features
- **Dataset**: Starbucks customer ordering patterns with 100,000+ records
- **Target Variable**: High-spend classification (binary: high spender or regular customer)
- **Model**: Random Forest Classifier with 100 estimators
- **Performance Metrics**: Accuracy and Confusion Matrix

## 📂 Project Structure

```
(Starbucks Dataset)/
├── dataset.py                          # Explore and analyze the dataset
├── preprocessing.py                    # Data loading and preprocessing functions
├── main.py                             # Main script to train the model
├── train model.py                      # Model training logic
├── prediction.py                       # Make predictions on new samples
├── test_debug.py                       # Testing and debugging utilities
├── starbucks_customer_ordering_patterns.csv  # Dataset
└── README.md                           # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Required packages (see Installation section)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/nayangit789/starbucks-customer-classification-ml.git
cd starbucks-customer-classification-ml
```

2. **Install required packages:**
```bash
pip install pandas scikit-learn
```

### Running the Project

1. **Explore the dataset:**
```bash
python dataset.py
```
This will display the first few rows, columns, data types, and missing values.

2. **Train the model:**
```bash
python main.py
```
This will:
- Load the dataset
- Preprocess the data (handle missing values, create target variable)
- Train a Random Forest classifier
- Display model accuracy and confusion matrix

3. **Make predictions (in Python):**
```python
from train model import train_model
from preprocessing import load_data, preprocess
from prediction import predict

# Load and train
df = load_data("starbucks_customer_ordering_patterns.csv")
x, y = preprocess(df)
model = train_model(x, y)

# Make prediction on a new sample
sample = [...]  # Feature values matching the trained model
result = predict(model, sample)
print("High Spender:" if result == 1 else "Regular Customer:", result)
```

## 📊 How It Works

### 1. Data Preprocessing (`preprocessing.py`)
- **Load Data**: Reads the CSV file using pandas
- **Handle Missing Values**: Removes rows with missing values
- **Create Target Variable**: 
  - Calculates median total spending
  - Classifies customers as high-spend (1) if spending >= median, else (0)
- **Feature Engineering**: 
  - Drops ID and timestamp columns
  - Applies one-hot encoding for categorical variables
- **Output**: Feature matrix (X) and target vector (Y)

### 2. Model Training (`train model.py`)
- **Algorithm**: Random Forest Classifier (100 estimators)
- **Train-Test Split**: 80-20 split with random_state=42
- **Evaluation Metrics**:
  - Accuracy Score
  - Confusion Matrix
- **Output**: Trained model ready for predictions

### 3. Making Predictions (`prediction.py`)
- Takes a trained model and feature sample
- Returns binary prediction (0 or 1)

## ⚙️ Technologies Used

- **Python 3**
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and evaluation metrics

## 📈 Model Performance

After training, the model displays:
- **Accuracy**: Percentage of correct predictions
- **Confusion Matrix**: True Positives, True Negatives, False Positives, False Negatives

Example output:
```
Accuracy: 0.82
Confusion Matrix:
 [[500  100]
  [ 80  320]]
```

## 🛠️ Troubleshooting

**Module not found error:**
```bash
pip install pandas scikit-learn
```

**CSV file not found:**
Make sure `starbucks_customer_ordering_patterns.csv` is in the same directory as the Python scripts.

**Train model import error:**
The file is named `train model.py` (with a space). This is imported using:
```python
import importlib
train_module = importlib.import_module("train model")
```

## 📌 Future Improvements

- [ ] Implement additional classifiers (Logistic Regression, SVM, XGBoost)
- [ ] Perform hyperparameter tuning
- [ ] Add cross-validation for better evaluation
- [ ] Feature importance analysis
- [ ] Create visualizations (ROC curves, feature importance plots)
- [ ] Build a web interface for predictions
- [ ] Add model persistence (save/load trained models)

## 👨‍💻 Author

**Nayan Git** - [GitHub](https://github.com/nayangit789)

## 📄 License

This project is open-source and available for educational purposes.

## 📞 Contact

For questions or contributions, feel free to open an issue or contact the repository owner.

---

**Last Updated**: March 2026
