# Customer Churn Prediction with a Neural Network (Deep Learning)

Predicts whether a telecom customer will churn using a feed-forward neural network built with Keras/TensorFlow.
This project builds on my earlier Power BI project, [Customer-Churn-Analysis](https://github.com/omjawale-oma/Customer-Churn-Analysis), which found that month-to-month contracts and customers in their first 6 months churn the most.

## Problem
Given a customer's demographics, account details, services and charges, predict `Churn` (Yes/No).

## Dataset
- `data/telco_churn_unclean.csv` - 7,048 rows, 21 columns (7,043 unique customers)
- About 26.5% of customers churned (imbalanced classes)

## Data cleaning
- Removed 5 duplicate rows
- Converted `MonthlyCharges` from text (with `$`) to numbers
- Fixed 134 blank/non-numeric `TotalCharges` values (tenure x monthly charges)
- Standardized `Contract` labels and trimmed extra spaces
- One-hot encoded categorical columns and standardized numeric features

## Model
Input -> Dense(64, ReLU) -> Dropout(0.3) -> Dense(32, ReLU) -> Dropout(0.3) -> Dense(16, ReLU) -> Dense(1, Sigmoid)

- Loss: binary cross-entropy, optimizer: Adam (lr 0.001)
- Class weights to handle imbalance
- Early stopping on validation loss (patience 10)
- 80/20 stratified train-test split

## Results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Neural network | 0.751 | 0.521 | 0.751 | 0.616 | 0.835 |
| Logistic regression | 0.742 | 0.509 | 0.789 | 0.618 | 0.841 |
| Random forest | 0.787 | 0.628 | 0.487 | 0.548 | 0.828 |

## Key takeaways
- The neural network correctly identified 75% of customers who churned (recall 0.75) with a ROC-AUC of 0.835.
- Logistic regression performed about the same (ROC-AUC 0.841), which is common on small tabular datasets (about 7,000 rows), where simple models are already very strong.
- Random forest had the highest accuracy (0.787) but missed more than half of the churners (recall 0.487), so it is less useful for retention campaigns.
- For churn, recall matters most, because missing a customer who will leave costs more than contacting one who would have stayed.
- These results fit the Power BI findings: customers on month-to-month contracts and with short tenure are the highest risk.

## How to run
```bash
pip install -r requirements.txt
python churn_nn.py
```
Use Python 3.10-3.12 (TensorFlow does not support newer versions yet).

## Author
**Om Jawale** - https://github.com/omjawale-oma
