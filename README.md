# **Medical Insurance Cost Prediction**

This repository contains a Jupyter Notebook that demonstrates how to predict medical insurance costs using machine learning techniques. The dataset used in this project includes demographic and health-related information of individuals.

---

## **Overview**

Predicting medical insurance costs is an essential task for healthcare providers and insurance companies to better understand the factors influencing healthcare expenses. This project uses **Linear Regression** to predict the insurance charges based on features such as age, BMI, number of children, smoking habits, and region.

The dataset includes demographic and health-related features, with the target variable (`charges`) representing the medical insurance cost.

---

## **Dataset**

- **Source**: The dataset appears to be related to publicly available health insurance datasets.
- **Features**:
  - `age`: Age of the individual.
  - `sex`: Gender of the individual (male/female).
  - `bmi`: Body Mass Index (BMI), a measure of body fat based on height and weight.
  - `children`: Number of children/dependents covered by health insurance.
  - `smoker`: Smoking status (yes/no).
  - `region`: Geographic region (e.g., southwest, southeast).
  - `charges`: Medical insurance cost (target variable).

---

## **Project Workflow**

1. **Data Loading**:
   - The dataset (`insurance.csv`) is loaded into a Pandas DataFrame.
2. **Exploratory Data Analysis (EDA)**:
   - Summary statistics and visualizations are generated using Seaborn and Matplotlib to explore relationships between features and the target variable.
3. **Data Preprocessing**:
   - Categorical variables such as `sex`, `smoker`, and `region` are encoded into numerical values for model training.
4. **Model Training**:
   - A Linear Regression model is trained to predict medical insurance costs.
   - The dataset is split into training and testing sets using `train_test_split`.
5. **Model Evaluation**:
   - Performance metrics such as Mean Absolute Error (MAE) and Mean Squared Error (MSE) are calculated to evaluate model accuracy.

---

## **Dependencies**

To run this project, you need the following Python libraries:

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn

You can install these dependencies using pip:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

---

## **How to Run**

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/MedicalInsuranceCostPrediction.git
   cd MedicalInsuranceCostPrediction
   ```

2. Ensure that the dataset file (`insurance.csv`) is in the same directory as the notebook.

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Med-Cost-Pred.ipynb
   ```

4. Run all cells in the notebook to execute the code.

---

## **Results**

The Linear Regression model predicts medical insurance costs based on input features like age, BMI, smoking status, etc. Evaluation metrics such as MAE and MSE indicate how well the model performs in predicting costs. Further improvements can be made by exploring other regression models or feature engineering techniques.

---

## **Acknowledgments**

- The dataset was sourced from publicly available health insurance datasets or repositories.
- Special thanks to Scikit-learn for providing robust machine learning tools.

---
