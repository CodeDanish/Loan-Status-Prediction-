Loan Status Prediction


Overview :

This repository contains the implementation of a machine learning model that predicts the status of a loan application as either "Approved" or "Rejected" based on various applicant features. The project aims to assist financial institutions in automating the loan approval process by accurately assessing the risk associated with each application.


Features:

- Data Preprocessing: Cleaned and preprocessed the dataset by handling missing values, encoding categorical variables, and standardizing numerical features.
- Exploratory Data Analysis (EDA): Performed EDA to uncover insights and relationships between features and the loan status.
- Feature Engineering: Created new features and selected the most relevant ones to improve model performance.
- Modeling: Experimented with various machine learning algorithms, including Logistic Regression, Decision Trees, Random Forest, and XGBoost, to identify the best-performing model.
- Model Evaluation: Evaluated model performance using accuracy, precision, recall, F1-score, and ROC-AUC curve.
- Deployment: Deployed the final model using Streamlit to create a user-friendly web application for real-time loan status prediction.


Installation:

To run this project on your local machine, follow these steps:

Clone the repository:

bash

Copy code

git clone https://github.com/yourusername/loan-status-prediction.git

Navigate to the project directory:

bash

Copy code

cd loan-status-prediction

Install the required dependencies:

bash

Copy code

pip install -r requirements.txt

Run the Streamlit application:

bash

Copy code

streamlit run app.py


Usage:

- Web Application: Use the web app to predict the status of a loan application by inputting the relevant applicant details, such as income, credit history, and loan amount. The app will output whether the loan is likely to be "Approved" or "Rejected."
- Notebooks: Explore the Jupyter notebooks in this repository to understand the data preprocessing, feature engineering, and model development process.


Technologies Used:

- Programming Language: Python
  
- Libraries:
  
Pandas: For data manipulation and analysis

Scikit-learn: For machine learning model development

XGBoost: For advanced gradient boosting techniques

Streamlit: For deploying the model as an interactive web application

Matplotlib/Seaborn: For data visualization and EDA


Contributing:

Contributions are welcome! If you would like to improve this project, please fork the repository, create a new branch, and submit a pull request. Ensure your contributions align with the project's objectives and follow best practices.


License:

This project is licensed under the MIT License. See the LICENSE file for more details.


Demo:

https://gudy2uptm5gvmjty9u74ue.streamlit.app/





