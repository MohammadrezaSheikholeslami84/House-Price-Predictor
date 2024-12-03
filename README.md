# House Price Predictor

## Description
This project predicts house prices based on various features such as square footage, number of bedrooms, location, and more. It uses machine learning models to train on historical housing data from Kaggle and make predictions for new properties. The goal is to provide a model that can estimate house prices based on given features.

## Table of Contents
- [Description](#description)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [License](#license)

## Technologies
This project is built using the following technologies:
- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- XGBoost

## Dataset
The dataset used in this project is the **[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)** dataset from Kaggle. The dataset contains the following features:
- **OverallQual**: Overall material and finish quality
- **GrLivArea**: Above ground living area square feet
- **BsmtQual**: Basement height
- **1stFlrSF**: First Floor square feet
- **YearBuilt**: Original construction date
- **TotalBsmtSF**: Total square feet of basement area
- **GarageCars**: Size of garage in car capacity
- **SalePrice**: The target variable, which is the sale price of the house

The dataset is provided in CSV format, and you can download it from Kaggle using the provided link.

## Model
The model is built using several regression algorithms, including:
- **Linear Regression**: A simple algorithm that models the relationship between input features and house prices.
- **Support Vector Regression (SVR)**: A more complex algorithm that uses support vectors to fit the model.
- **XGBoost**: A gradient boosting algorithm known for high performance and scalability.
- **K-Nearest Neighbors Regressor**: A non-parametric algorithm that predicts house prices based on the k-nearest data points.
- **Decision Tree Regressor**: A tree-based model that splits data into branches based on feature values.

### Steps:
1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numeric features.
2. **Model Training**: Training multiple regression models and selecting the best one based on evaluation metrics.
3. **Evaluation**: Evaluating the model performance using metrics such as **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R² score**.

## Results
The model's performance is evaluated using the test data. You can visualize the results using various plots like:
- Actual vs Predicted house prices
- Error distribution

The models provide predictions of house prices with reasonable accuracy, and further optimizations can be made by fine-tuning hyperparameters or experimenting with other machine learning algorithms.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/MohammadrezaSheikholeslami84/House-Price-Predictor.git
   cd House-Price-Predictor
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Train the Model**:  
   To train the machine learning models, run:
   ```bash
   python train_model.py
   ```

2. **Make Predictions**:  
   Once the model is trained, you can use it to make predictions on new data:
   ```bash
   python predict.py --input <path_to_new_data.csv>
   ```

3. **Visualize Results**:  
   To visualize the training and prediction results, you can generate plots using:
   ```bash
   python visualize_results.py
   ```


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
