# Customer Churn Prediction using Artificial Neural Network (ANN)

## Project Overview

This project predicts whether a customer will churn or not using an artificial neural network (ANN) model. Churn prediction is essential for identifying customers who are likely to stop using a service, allowing companies to take proactive measures to retain them. The project is deployed using **Streamlit**, making it easy for users to interact with the model through a simple web interface.

You can access the deployed model here: [Customer Churn Prediction App](https://churning.streamlit.app/).

## Features

- **Data Processing:** The dataset used contains various features such as customer demographics, account details, and transaction history.
- **Modeling:** An artificial neural network (ANN) is used to predict customer churn based on the input features.
- **Deployment:** The model is deployed using **Streamlit** for easy interaction and visualization.

## Dataset

The dataset contains information about customers, including:
- Customer ID
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary
- Exited (Target Variable indicating churn)

## Model

The artificial neural network (ANN) model consists of the following:
- **Input Layer:** Receives input features.
- **Hidden Layers:** Multiple layers with neurons to learn patterns from the data.
- **Output Layer:** Provides the final binary prediction (Churn: Yes/No).

The model is trained using backpropagation with the **Adam optimizer** and **binary cross-entropy loss** function.

## Streamlit Deployment

The project is deployed using **Streamlit**, which provides an interactive user interface. Users can input customer data through the app, and the ANN model will predict whether the customer is likely to churn.

### Steps to run the project locally:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/SudarshanNK21/Churn-using-ANN-
    cd churn-prediction-ann
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Installation Requirements

- **Python 3.8+**
- **TensorFlow/Keras** for building and training the ANN
- **Pandas** and **NumPy** for data manipulation
- **Scikit-learn** for data preprocessing and evaluation
- **Streamlit** for deployment
- **Matplotlib** or **Seaborn** for data visualization

## How to Use the App

1. Open the [Churn Prediction App](https://churning.streamlit.app/).
2. Input the customer data (e.g., age, gender, balance, etc.).
3. Click on the **Predict** button to see the model's prediction on whether the customer is likely to churn.

## Results

The ANN model provides an accuracy score of 89%. The performance metrics are evaluated using:
- **Confusion Matrix**
- **Precision**
- **Recall**
- **F1 Score**

## Future Enhancements

- Improve model performance with hyperparameter tuning.
- Add more features to the dataset to improve prediction accuracy.
- Integrate the model with a real-time customer database.
