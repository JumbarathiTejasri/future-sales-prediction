
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def load_data():
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv"
    return pd.read_csv(url)

def train_model(data):
    x = np.array(data.drop(["Sales"], axis=1))
    y = np.array(data["Sales"])
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(xtrain, ytrain)

    score = model.score(xtest, ytest)
    return model, score

def predict_sales(model, features):
    return model.predict([features])[0]
