

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
# TODO: Import all the linear regression models that you used here
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

# LOADING DATA
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/10/data.csv"
)

"""
# 1985 Auto Imports Database

Abstract: This data set consists of three types of entities: 
        (a) the specification of an auto in terms of various characteristics, 
        (b) its assigned insurance risk rating, 
        (c) its normalized losses in use as compared to other cars. The second rating 
        corresponds to the degree to which the auto is more risky than its price indicates. 
        Cars are initially assigned a risk factor symbol associated with its
        price.   Then, if it is more risky (or less), this symbol is
        adjusted by moving it up (or down) the scale.  Actuarians call this
        process "symboling".  A value of +3 indicates that the auto is
        risky, -3 that it is probably pretty safe.
"""

@st.cache_data
def load_data(nrows):
    # TODO: Import the data, preprocess it, and seperate it into training and testing data
    df = pd.read_csv('data.csv')
    df.dropna(inplace=True)  
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop('symboling', axis=1)  
    t = df['symboling']  
    
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2, random_state=42)
    
    return df, X_train, X_test, t_train, t_test

df, X_train, X_test, t_train, t_test = load_data(1000)



"## Summary"    
st.dataframe(df.describe())

def evaluate(model, X_test, t_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(t_test, predictions)
    r2 = r2_score(t_test, predictions)
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, f'MSE: {mse:.2f}\nR²: {r2:.2f}', fontsize=12, ha='center')
    ax.axis('off')
    return fig


def show_weights(model, feature_names):
    weights = model.coef_
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(feature_names, weights)
    ax.set_xlabel('Weight')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Weights')
    plt.tight_layout()
    return fig


model_option = st.selectbox("Select model", ["Linear Regression", "Ridge", "Lasso", "ElasticNet", "SGDRegressor"])

if model_option == "Linear Regression":
    model = LinearRegression()
elif model_option == "Ridge":
    alpha = st.slider("alpha", 0.01, 1.0, 0.1)
    model = Ridge(alpha=alpha)
elif model_option == "Lasso":
    alpha = st.slider("alpha", 0.01, 1.0, 0.1)
    model = Lasso(alpha=alpha)
elif model_option == "ElasticNet":
    alpha = st.slider("alpha", 0.01, 1.0, 0.1)
    l1_ratio = st.slider("l1_ratio", 0.0, 1.0, 0.5)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
elif model_option == "SGDRegressor":
    alpha = st.slider("alpha", 0.01, 1.0, 0.1)
    learning_rate = st.selectbox("Learning Rate", ["constant", "optimal", "invscaling", "adaptive"])
    model = SGDRegressor(alpha=alpha, learning_rate=learning_rate)

model.fit(X_train, t_train)

"## Model Evaluation"
st.pyplot(evaluate(model, X_test, t_test))

"## Model Weights"
st.pyplot(show_weights(model, df.columns[:-1]))  
