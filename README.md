# Hotel Booking Cancellation Prediction

This project aims to predict whether a hotel booking will be canceled or not using machine learning techniques. The dataset includes information about hotel bookings, and the goal is to build a model that can accurately predict cancellations.

## Project Overview

The project is divided into several steps:

1. **Data Loading**: Load the training dataset (`hotels_train.csv`).
2. **Exploratory Data Analysis (EDA)**: Visualize and understand the data.
3. **Data Cleaning**: Handle missing or problematic data.
4. **Data Balancing**: Use SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
5. **Modeling**: Train and evaluate multiple machine learning models.
6. **Model Evaluation**: Use ROC curve and T-test to compare models.
7. **Prediction**: Preprocess the test dataset (`hotels_test.csv`), apply the best model, and save the results to `results.csv`.

## Files in the Repository

- `hotel_booking_cancellation.ipynb`: Jupyter Notebook containing the project code.
- `hotels_train.csv`: Training dataset.
- `hotels_test.csv`: Test dataset.
- `results.csv`: Results from the best model on the test dataset.

## Getting Started

### Prerequisites

Ensure you have the following Python packages installed:
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn

You can install these packages using pip:
'pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn'

##Running the Project

    Clone this repository to your local machine.
    Open the Jupyter Notebook hotel_booking_cancellation.ipynb.
    Follow the steps in the notebook to see the data loading, EDA, data cleaning, data balancing, model training, and evaluation processes.
    Apply the best model to the test dataset and save the predictions to results.csv.

## Project Steps

### Data Loading

The training data is loaded from `hotels_train.csv` and the test data is loaded from `hotels_test.csv`.

### Exploratory Data Analysis (EDA)

Visualizations and summary statistics are used to understand the distribution and relationships within the data.

### Data Cleaning

Handling missing values and correcting data issues to ensure the dataset is ready for modeling.

### Data Balancing

Using SMOTE to balance the dataset, ensuring the model does not favor one class over the other.

### Modeling

Several machine learning models are trained and evaluated, including:
- Decision Tree
- Random Forest
- Gaussian Naive Bayes
- Support Vector Machine (SVM)
- Gradient Boosting
- AdaBoost

### Model Evaluation

Models are compared using ROC curves and T-tests to determine the best-performing model.

### Prediction

The test dataset is preprocessed to match the training data. The best model is applied to the test data, and the predictions are saved to `results.csv`.

### Results

The predictions from the best model are saved in `results.csv`. This file contains the model's predictions on whether each booking in the test dataset will be canceled or not.

### Conclusion

This project demonstrates the end-to-end process of building a machine learning model to predict hotel booking cancellations. The final model can be used to assist hotels in managing their bookings more effectively.








