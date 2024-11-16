import pickle

import pandas as pd

from sklearn.linear_model import LinearRegression


data: pd.DataFrame = pd.read_csv("files/input/house_data.csv")
features: pd.DataFrame = data[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]
target = data["price"]
linear_model = LinearRegression()
linear_model.fit(features, target)

with open("homework/house_predictor.pkl", "wb") as file:
    pickle.dump(linear_model, file)
