from flask import Flask, render_template, request
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from Models import Calories_Model as cm
from Models import Diabetes_Model as dm
from Models import Heart_Model as hm

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load the pickle model here
calories_model = pickle.load(open("./Models/Calories_Model.pkl", "rb"))
diabetes_model = pickle.load(open("./Models/Diabetes_Model.pkl", "rb"))
heart_model = pickle.load(open("./Models/Heart_Model.pkl", "rb"))


@app.route('/')
def home():
    return render_template('Home.html')
    # return '<h1>Home<h1/>'


@app.route("/prediction/calories", methods=["POST"])
def predict():
    # Get the values from the form and convert them to float
    gender = int(request.form.get("genderValue"))
    age = int(request.form.get("ageValue"))
    height = float(request.form.get("heightValue"))
    weight = float(request.form.get("weightValue"))
    duration = float(request.form.get("durationValue"))
    heart_rate = float(request.form.get("heartRateValue"))
    body_temp = float(request.form.get("bodyTemperatureValue"))

    # features = [gender, age, height, weight, duration, heart_rate, body_temp]
    # for idx, value in enumerate(features):
    #     print(f"{idx+1} = {value}")

    model = LinearRegression()

    prediction = cm.calories_prediction(ml_model=model, gender=gender, age=age, height=height, weight=weight,
                                        duration=duration,
                                        heart_rate=heart_rate, body_temp=body_temp)
    return render_template("predictionResult.html", prediction_text="Calories Burnt Value = {} kcal".format(prediction), model_text = "Calories Burnt Prediction", accuracy_text = "96% Accuracy")

@app.route("/prediction/diabetes", methods=["POST"])
def pred_diabetes():
    # Getting the values here
    pregnancy = int(request.form.get("pregValue"))
    age = int(request.form.get("ageValue"))
    glucose = float(request.form.get("glucoseValue"))
    skin = float(request.form.get("skinValue"))
    insulin = float(request.form.get("insulinValue"))
    bmi = float(request.form.get("bmiValue"))
    dpf = float(request.form.get("dpfValue"))
    bp = float(request.form.get("bpValue"))

    # features = [pregnancy, age, glucose, skin, insulin, bmi, dpf, bp]
    # for idx, value in enumerate(features):
    #     print(f"{idx + 1} = {value}")

    model = LogisticRegression()

    prediction = dm.diabetes_prediction(ml_model=model, pregnancies=pregnancy, age=age, glucose=glucose, st=skin,
                                        insulin=insulin,
                                        bmi=bmi, dpf=dpf, bp=bp)

    return render_template("predictionResult.html",
                           prediction_text="{}".format(prediction), model_text = "Diabetes Prediction Result", accuracy_text = "80% Accuracy")


@app.route('/prediction/heart', methods=["POST"])
def pred_heart():
    # fetching all the values here
    gender = int(request.form.get("genderValue"))
    age = int(request.form.get("ageValue"))
    cp = int(request.form.get("cpValue"))
    trest = int(request.form.get("trestbpsValue"))
    cholestrol = int(request.form.get("cholestrolValue"))
    fbs = int(request.form.get("fbsValue"))
    restecg = int(request.form.get("restecgValue"))
    thalach = int(request.form.get("thalachValue"))
    exang = int(request.form.get("exangValue"))
    oldpeak = float(request.form.get("oldpeakValue"))
    slope = int(request.form.get("slopeValue"))
    ca = int(request.form.get("caValue"))
    thal = int(request.form.get("thalValue"))

    # features = [gender, age, cp, trest, cholestrol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    # for idx, value in enumerate(features):
    #     print(f"{idx+1}. {value}")

    model = LogisticRegression()

    prediction = hm.predict_heart(ml_model=model, age=age, gender=gender, cp=cp, trestbps=trest, chol=cholestrol, fbs=fbs, restecg=restecg, thalach=thalach, exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)

    return render_template("predictionResult.html",
                           prediction_text="{}".format(prediction), model_text="Heart Disease Prediction Result",
                           accuracy_text="85.25% Accuracy")


@app.route('/CaloriesPredictor')
def caloriesPage():
    return render_template('CaloriesPredictor.html')
    # return '<h1>Calories Page<h1/>'


@app.route('/HeartDiseasePredictor')
def heartPage():
    return render_template('HeartDiseasePredictor.html')
    # return '<h1>Disease Page<h1/>'


@app.route('/DiabetesPredictor')
def diabetesPage():
    return render_template('DiabetesPredictor.html')
    # return '<h1>Diabetes Page<h1/>'


if __name__ == '__main__':
    app.run(debug=False, port=8000)
