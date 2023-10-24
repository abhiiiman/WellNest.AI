# import required libraries
import pickle

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from warnings import filterwarnings

filterwarnings("ignore")

sns.set()

# Load the Calories dataset
# df1 = pd.read_csv("../Datasets/calories.csv")
df1 = pd.read_csv(r"C:\Users\DELL\PycharmProjects\pythonProject\Datasets\calories.csv")

# Load the Exercise Dataset
# df2 = pd.read_csv("../Datasets/exercise.csv")
df2 = pd.read_csv(r"C:\Users\DELL\PycharmProjects\pythonProject\Datasets\exercise.csv")

# Now Concatenate both the Dataframe i.e. df1 and df2
df = pd.concat([df2, df1["Calories"]], axis=1)

# drop User_ID column because this is not required from Main Dataframe itself
df.drop(columns=["User_ID"], axis=1, inplace=True)

# Separate Categorical and Numerical Features
# 1. Categorical Feature

# Fetching Categorical Data
cat_col = [col for col in df.columns if df[col].dtype == 'O']  # -->Object-"o"

categorical = df[cat_col]
categorical = pd.get_dummies(categorical["Gender"], drop_first=True)

# 2. Numerical Features
Num_col = [col for col in df.columns if df[col].dtype != "O"]
Numerical = df[Num_col]

# Concatenate Categorical and Numerical
data = pd.concat([categorical, Numerical], axis=1)

# Storing Independent and Dependent features
X = data.drop(columns=["Calories"], axis=1)
y = data["Calories"]

# Performing the train and test split here
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# from sklearn import metrics
def calories_predict_summary(ml_model):
    model = ml_model.fit(X_train, y_train)
    print('Score : {}'.format(model.score(X_train, y_train)))
    y_prediction = model.predict(X_test)
    print('predictions are: \n {}'.format(y_prediction))
    print('\n')

    r2_score = metrics.r2_score(y_test, y_prediction)
    print('r2 score: {}'.format(r2_score))

    print('MAE:', metrics.mean_absolute_error(y_test, y_prediction))
    print('MSE:', metrics.mean_squared_error(y_test, y_prediction))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_prediction)))

    sns.distplot(y_test - y_prediction)


def calories_prediction(ml_model, gender, age, height, weight, duration, heart_rate, body_temp):
    model = ml_model.fit(X_train, y_train)
    y_prediction = model.predict([[gender, age, height, weight, duration, heart_rate, body_temp]])
    return y_prediction[0]


# Linear Regression prediction summary
# calories_predict_summary(LinearRegression())

# Making pickle file for our model
pickle.dump(LinearRegression, open("Calories_Model.pkl", "wb"))
