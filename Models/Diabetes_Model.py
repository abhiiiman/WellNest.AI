# import required libraries
import pickle
import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from warnings import filterwarnings

filterwarnings("ignore")

# Loading the Calories dataset
df = pd.read_csv(r"C:\Users\DELL\PycharmProjects\pythonProject\Datasets\diabetes.csv")

# outlier remove

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# print("---Q1--- \n", Q1)
# print("\n---Q3--- \n", Q3)
# print("\n---IQR---\n", IQR)

# outlier remove
df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# extracting features and targets
X = df_out.drop(columns=['Outcome'])
y = df_out['Outcome']

# Splitting train test data 80 20 ratio
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)


def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]


def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]


# cross validation purpose
scoring = {'accuracy': make_scorer(accuracy_score), 'prec': 'precision'}
scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
           'fp': make_scorer(fp), 'fn': make_scorer(fn)}


def display_result(result):
    print("TP: ", result['test_tp'])
    print("TN: ", result['test_tn'])
    print("FN: ", result['test_fn'])
    print("FP: ", result['test_fp'])


# building the model here

# Logistic Regression
acc = []
roc = []

model = LogisticRegression()
model.fit(train_X, train_y)
y_pred = model.predict(test_X)

# find accuracy
ac = accuracy_score(test_y, y_pred)
acc.append(ac)

# find the ROC_AOC curve
rc = roc_auc_score(test_y, y_pred)
roc.append(rc)
# print("\nAccuracy {0} ROC {1}".format(ac, rc))

# cross val score
result = cross_validate(model, train_X, train_y, scoring=scoring, cv=10)
# display_result(result)


def diabetes_prediction(ml_model, pregnancies, glucose, bp, st, insulin, bmi, dpf, age):
    model = ml_model.fit(train_X, train_y)
    y_prediction = model.predict([[pregnancies, glucose, bp, st, insulin, bmi, dpf, age]])
    if y_prediction == 1:
        return "You have Diabetes."
    else:
        return "You're safe, you don't have Diabetes."


# features.
# Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

# creating the model constructor here.
my_model = LogisticRegression()

# Making pickle file for our model
pickle.dump(LogisticRegression, open("Diabetes_Model.pkl", "wb"))
