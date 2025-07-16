# PRODIGY_DS_03
Predict customer subscription to a term deposit using a Decision Tree Classifier trained on the UCI Bank Marketing dataset, with automatic data fetching and visualization.
# 🌳 Bank Marketing Prediction with Decision Tree Classifier

This project builds a machine learning model using a Decision Tree to predict whether a customer will subscribe to a term deposit based on their demographic and marketing interaction data.

✅ The dataset is automatically fetched from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) using the `ucimlrepo` package.

---

## 🧠 Features Used

The dataset includes customer information like:

- Age, Job, Marital Status, Education
- Contact Method, Day, Duration
- Previous Campaign Outcomes
- Target variable: `y` (Yes/No) – did the client subscribe?

---

## 📦 Tech Stack

| Tool           | Purpose                      |
|----------------|------------------------------|
| `pandas`       | Data handling                |
| `scikit-learn` | ML modeling + preprocessing  |
| `matplotlib`   | Plotting the decision tree   |
| `seaborn`      | Confusion matrix heatmap     |
| `ucimlrepo`    | Auto-fetching UCI dataset    |

---

## 🧪 How to Run

1. 🔧 Install dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn ucimlrepo
