import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
import pickle

warnings.filterwarnings('ignore')

# importing the dataset here
dataset = pd.read_csv(r"C:\Users\DELL\PycharmProjects\pythonProject\Datasets\heart.csv")

# Train Test Split here
predictors = dataset.drop("target", axis=1)
target = dataset["target"]
# Splitting the data here in 80:20
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

# Model fitting
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred_lr = lr.predict(X_test)
# print(Y_pred_lr)


# creating the method for prediction here
def predict_heart(ml_model, age, gender, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    model = ml_model.fit(X_train, Y_train)
    y_prediction = model.predict([[age, gender, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    if y_prediction == 1:
        return "You have Heart Disease."
    else:
        return "You're safe, you don't have Heart Disease."


# score - accuracy
score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)
# print("The accuracy score achieved using Logistic Regression is: " + str(score_lr) + " %")

# creating the model constructor here.
my_model = LogisticRegression()

# Making pickle file for our model
pickle.dump(LogisticRegression, open("Heart_Model.pkl", "wb"))
