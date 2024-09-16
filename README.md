# Medical Insurance Cost Prediction
## Overview
This project demonstrates a predictive model that estimates medical insurance costs based on various factors like age, gender, BMI, number of children, smoking status, and region. The model is built using Linear Regression and is evaluated using the R-squared metric for both training and testing sets.

## Prerequisites
To run this code, ensure you have the following Python libraries installed:
* numpy
* pandas
* matplotlib
* seaborn
* scikit-learn

You can install these libraries using pip:

> pip install numpy pandas matplotlib seaborn scikit-learn

## Dataset
The dataset used in this project is insurance.csv, which includes the following columns:

- age: Age of the insured
- sex: Gender (male/female)
- bmi: Body Mass Index (BMI)
- children: Number of children/dependents
- smoker: Smoking status (yes/no)
- region: Residential region
- charges: Insurance charges

## Data Processing
- **Exploratory Data Analysis (EDA):** Visualizes the distribution of each feature using seaborn and matplotlib.
- **Data Encoding:** Categorical variables like sex, smoker, and region are encoded into numerical values using the replace function.

## Model
The code uses Linear Regression from the sklearn.linear_model library to predict insurance costs.
- The features (X) are the input variables except charges.
- The target (Y) is the charges column, which represents the cost of insurance.
- The dataset is split into training (80%) and testing (20%) sets using the train_test_split function.

## Model Evaluation
- The model's performance is evaluated using the R-squared metric (metrics.r2_score) for both training and testing sets.
- Training R²: Reflects how well the model fits the training data.
- Testing R²: Reflects how well the model generalizes to unseen data.

## Prediction
A sample input (input_data) is provided to predict insurance costs using the trained model. For example:

> input_data = (31, 1, 25.74, 0, 1, 0)

- The model predicts the insurance cost based on the given input features.

## How to Run
- Ensure the insurance.csv file is present in your working directory.
- Run the script in your Python environment.
- Check the output for the model’s performance on training and test sets, along with a prediction for sample input.

## Results
- The output displays the R-squared score for both training and testing sets.
- A sample prediction for insurance cost is made using predefined input data.

For Example: 
> Training set score: 0.7515
> 
> Test set score: 0.744
> 
> The insurance cost in USD: 3760.20
