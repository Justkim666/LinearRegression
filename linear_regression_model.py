import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import joblib

# Read in the dataset
df = pd.read_csv("./data.csv")
df.columns = ["index", "height", "weight"]
print(df.head())

# get x and y
x = df["height"].values
y = df["weight"].values

# find m and b (OLS - Ordinary Least Squares)
# Linear Regression: y = m * x_new + b
N = x.shape[0] # return number of elements in x 
m = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x ** 2) -  (np.sum(x) ** 2))
b = (np.sum(y) - m * np.sum(x)) / N
print(f"Simple Linear Regression (SLR) \nm = {m:.2f} \nb = {b:.2f}")
joblib.dump((m, b), "linear_regression_model_manual.pkl")

# model
x_reshaped = x.reshape(-1, 1)

model = LinearRegression()
model.fit(x_reshaped, y)

print(f"Simple Linear Regression (sklearn) \nm = {model.coef_[0]:.2f} \nb = {model.intercept_:.2f}")

joblib.dump(model, "linear_regression_model.pkl")

# Draw regression line
x_min = np.min(x)
y_min = m * x_min + b
x_max = np.max(x)
y_max = m * x_max + b

fig, ax = plt.subplots()
sns.scatterplot(
    data=df,
    x="height",
    y ="weight",
    ax=ax
)
sns.lineplot(
    x=[x_min, x_max],
    y=[y_min, y_max],
    color='red'
)
plt.show()