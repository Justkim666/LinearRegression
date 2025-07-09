import pandas as pd
import numpy as np

num_sample = 2000

heights = np.random.normal(loc=66, scale=3, size=num_sample)

weights = heights * 2.3 + np.random.normal(loc=0, scale=10, size=num_sample)

data = pd.DataFrame({
    "Index": np.arange(1, num_sample + 1),
    "Height(Inches)": np.round(heights, 2),
    "Weights(Pounds)": np.round(weights, 2)
})

data.to_csv("data.csv", index=False)