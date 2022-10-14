import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_housing_data():
    return pd.read_csv('housing.csv')
  
housing = load_housing_data()

housing["income_cat"] = pd.cut(housing["median_income"],
bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
labels=[1, 2, 3, 4, 5])

housing["income_cat"].hist()
plt.show()

print(housing.info())
# housing.to_csv('housing_new.csv')