# IBM HR Analytics Project

## Description
This project predicts employee attrition using HR data and machine learning.
It includes:
- Exploratory Data Analysis (EDA)
- Random Forest model training
- Hyperparameter tuning
- Feature importance visualization

## Folder Structure

Ibm_Hr_Project/
│
├── data/                      # Dataset files (.csv)
│
├── plots/                     # Saved visualizations (PNG charts)
│   ├── Attrition_count_plot.png
│   ├── attrition_plot.png
│   ├── correlation_heatmap.png
│   └── feature_importance.png
│
├── scripts/                   # All Python scripts
│   ├── dashboard_hr.py        # Streamlit dashboard
│   ├── eda.py                 # Exploratory Data Analysis code
│   ├── model.py               # ML model training + tuning
│   ├── utils.py               # Helper functions
│   └── main.py                # Main execution script
│
├── README.md                  # Project documentation
├── requirements.txt           # Libraries used
└── .gitignore                 # Files to ignore in Git
Clone the repository

git clone https://github.com/akhilamudhiraj/Ibm_Hr_Project.git
Navigate to the folder

cd Ibm_Hr_Project
Create virtual environment

python -m venv venv
Activate virtual environment
Windows (PowerShell):

.\venv\Scripts\Activate.ps1
Install dependencies

pip install -r requirements.txt
Run the HR Dashboard
streamlit run scripts/dashboard_hr.py
Run EDA File
python scripts/eda.py

Run the Model Training Script
python scripts/model.py
Project Features

📊 Interactive HR Analytics Dashboard (Streamlit)

👥 Attrition breakdown by age, salary, job role, department

🔥 Machine Learning Model: Random Forest

📈 Feature importance visualization

🔍 Exploratory Data Analysis (EDA)

📉 Correlation heatmap

🎯 Predictive analytics for employee attrition

🚀 Clean folder structure with modular scripts
Results & Visualizations

This project includes several visual insights that help understand employee attrition trends:

Attrition Count Plot – Shows the number of employees who stayed vs. left

Age vs Attrition Plot – Highlights which age groups are more likely to leave

Correlation Heatmap – Shows relationships between HR variables

Feature Importance Plot – Displays which features impact attrition the most

Department / Job Role Analysis – Visual insights on attrition by department

All plots are stored in:

plots/
Conclusion

This IBM HR Analytics project provides a complete workflow from data exploration to model building and visualization.
The dashboard and machine learning model help HR teams understand:

Why employees leave

Which factors influence attrition the most

How to improve retention strategies

Early identification of employees at risk

This project can be extended and deployed for real-time HR analytics in organizations.
