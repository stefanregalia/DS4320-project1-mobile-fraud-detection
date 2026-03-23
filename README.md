# DS4320-project1-mobile-fraud-detection

## Problem Definition

**Initial General Problem**:

As digital payments are becoming increasingly more common, cases of financial fraud are growing rapidly, impacting the integrity and safety of financial institutions and consumers, while costing both groups billions of dollars.

**Refined Specific Problem**:

Can we predict whether a mobile payment is fraudulent based on transaction type, amount, whether the recipient is the merchant or customer, the hour of the day, and the day of the month?

The general problem of financial fraud was refined to focus on predicting fraudulent activity in mobile payments using the PaySim synthetic dataset. This dataset simulates one month of mobile transactions, describing transaction types, amounts, recipient type, the time of day, and the day of the month. We use synthetic data for this refined problem, as real-world financial fraud datasets are either anonymized or not publicly available, and are therefore, difficult to interpret and make predictions. We exclude variables describing the before and after transaction monetary balance for each customer to prevent data leakage, as the simulation process causes balance variables to reflect the outcome of fraudulent transactions instead of treating them as independent predictors.

The motivation behind this project is to detect fraudulent mobile money transactions so that we can identify suspicious transactions and prevent financial losses from occurring. Tens of thousands of fraudulent mobile transactions occur per day, costing customers and institutions millions, or even billions of dollars a year. By identifying trends in these mobile transactions, such as common fraudulent transaction types or unusually large amounts, institutions can warn customers about potential scams. Institutions can also implement preventive measures such as identity and transaction verification systems to reduce the likelihood of fraudulent activity.

[Machine Learning Model Detects Fraudulent Mobile Transactions](press_release.md)


## Domain Exposition


| Term | Definition |
|------|------------|
| Mobile Money (Contextual) | A financial service that allows users to store, send, and receive money using a mobile phone |
| PaySim (Contextual) | A mobile money transaction simulator that generates synthetic financial data based on real transaction logs |
| isFraud (Original Feature) | Binary label indicating whether a transaction is fraudulent (1) or legitimate (0) |
| type (Original Feature) | The category of a transaction: CASH-IN, CASH-OUT, DEBIT, PAYMENT, or TRANSFER |
| amount (Original Feature) | The monetary value of the transaction in local currency |
| step (Original Feature) | A unit of simulated time where each step represents one hour; the dataset spans 743 steps (31 days) |
| nameOrig (Original Feature) | The unique identifier of the customer initiating the transaction |
| nameDest (Original Feature) | The unique identifier of the recipient of the transaction |
| is_merchant (Feature Engineered) | A binary flag indicating whether the recipient is a merchant (nameDest starts with M) or a customer (nameDest starts with C) |
| hour_of_day (Feature Engineered) | The hour within a 24-hour cycle, derived by computing step modulo 24 |
| day_of_month (Feature Engineered) | The day within the 31-day simulation period, derived by computing step divided by 24 |
| Class Imbalance (Contextual) | A condition where one class is significantly underrepresented relative to the other|
| Precision (KPI) | The proportion of predicted fraudulent transactions that are actually fraudulent |
| Recall (KPI) | The proportion of actual fraudulent transactions that the model correctly identifies |
| F1 Score (KPI) | The harmonic mean of precision and recall, used as the primary evaluation metric due to class imbalance |
| Binary Classification (Contextual) | A machine learning task where each transaction is classified into one of two categories: fraudulent or legitimate |
| Data Leakage (Contextual) | The unintentional inclusion of information in model training that would not be available at prediction time, such as balance variables in this dataset |

This project lives at the intersection between the domains of financial technology, mobile payment systems, and machine learning. Mobile payment systems, such as Venmo and Zelle, are platforms where people can send money to others directly through their phones. Financial technology companies that run these platforms implement systems to track millions of these mobile transactions every day, which would be impossible to do manually. Machine learning solves this problem by identifying patterns in fraudulent activity so that these companies can mitigate fraud and take as many preventative actions as possible.

Link to Google Drive Folder with articles:

https://drive.google.com/drive/folders/1DJX2ydABfxP4MmT6pTRmrynw6SaiV1Qn?usp=sharing

| Title | Description | Link |
|-------|-------------|------|
| PaySim: A Financial Mobile Money Simulator for Fraud Detection | Presents the PaySim simulator, describing how it generates synthetic mobile money transaction data based on real financial logs and injects fraudulent behavior for fraud detection research | https://github.com/stefanregalia/DS4320-project1-mobile-fraud-detection/blob/main/background_reading/PaySim.pdf |
| Fraud Prevention: An Overview | Provides a broad overview of fraud prevention strategies, explaining the types of financial fraud and how institutions detect and respond to them | https://github.com/stefanregalia/DS4320-project1-mobile-fraud-detection/blob/main/background_reading/Fraud%20prevention_%20An%20overview.pdf |
| Fintech Fraud Detection: Techniques, Tools & Solutions | Discusses fraud detection methods used in financial technology, covering common fraud types in mobile payment systems and the tools companies use to combat them | https://github.com/stefanregalia/DS4320-project1-mobile-fraud-detection/blob/main/background_reading/Comprehensive%20Guide%20to%20Fintech%20Fraud%20Detection%20(2025).pdf |
| Balancing the Scales: A Comprehensive Study on Tackling Class Imbalance in Binary Classification | Evaluates strategies for handling class imbalance in binary classification tasks, comparing SMOTE, class weights, and decision threshold calibration across multiple models and datasets | https://github.com/stefanregalia/DS4320-project1-mobile-fraud-detection/blob/main/background_reading/Class_imbalance.pdf |
| Fraud Detection in Mobile Payment Systems using an XGBoost-based Framework | Proposes an XGBoost-based framework for detecting fraud in mobile payment systems | https://github.com/stefanregalia/DS4320-project1-mobile-fraud-detection/blob/main/background_reading/Fraud%20Detection%20in%20Mobile%20Payment%20Systems%20using%20an%20XGBoost-based%20Framework%20-%20PMC.pdf |