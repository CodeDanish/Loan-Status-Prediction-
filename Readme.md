
# Loan Status Prediction 🏦💸

![Loan Status](https://img.shields.io/badge/Loan-Prediction-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 📚 Overview

The **Loan Status Prediction** project is aimed at predicting whether a loan will be approved or rejected based on key factors like applicant income, loan amount, credit history, and more. The project demonstrates the process of **data cleaning**, **feature engineering**, **model building**, and **model evaluation** using various machine learning algorithms.

### 🎯 Objective
To build a robust machine learning model that can accurately predict loan status for applicants.

## 📂 Project Structure

```
loan-status-prediction/
├── data/
│   ├── train.csv             # Training dataset
│   ├── test.csv              # Test dataset
├── notebooks/
│   ├── data_analysis.ipynb   # EDA and data analysis
│   ├── model_building.ipynb  # Model development and evaluation
├── app.py                    # Streamlit app for loan status prediction
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🚀 Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/loan-status-prediction.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd loan-status-prediction
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## ⚙️ Model Building Process

### 1. Exploratory Data Analysis (EDA) 📊
The first step is to understand the data, visualize it, and perform necessary cleaning operations.

### 2. Feature Engineering 🛠️
Selected key features like `ApplicantIncome`, `LoanAmount`, `Credit_History` to train the machine learning models.

### 3. Model Training 🤖
Several machine learning models were trained and evaluated:
- **Logistic Regression**
- **Random Forest**
- **XGBoost**

### 4. Model Evaluation 🏅
The best model achieved:
- **Accuracy:** 85%
- **Precision:** 0.87
- **Recall:** 0.84
- **F1-Score:** 0.85

## 🖥️ Demo

You can interact with the loan prediction model using the **Streamlit app**:

https://gudy2uptm5gvmjty9u74ue.streamlit.app/

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas, Numpy** for data analysis
- **Scikit-learn** for machine learning
- **XGBoost** for advanced model tuning
- **Streamlit** for the web interface

## 🤝 Contributing

Contributions are **welcome**! If you find any issues or have suggestions, feel free to open an issue or submit a pull request. For major changes, please open an issue first to discuss the proposed changes.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> 🔗 **References**:
> - [Pandas Documentation](https://pandas.pydata.org/)
> - [Scikit-learn Documentation](https://scikit-learn.org/stable/)
> - [Streamlit Documentation](https://docs.streamlit.io/)

