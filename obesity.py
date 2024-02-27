from flask import Flask, render_template,request
import joblib
import numpy as np 


app = Flask(__name__)

model = joblib.load("obesity_model.joblib")

weight_categories = {
    0: 'Underweight',
    1: 'Normal weight',
    2: 'Overweight',
    3: 'Obese'}

@app.route('/')
def homepage():
    return render_template('obesity.html')

@app.route('/predict', methods= ['POST'])
def predict():
    age=int(request.form['age'])
    gender = int(request.form['gender'])
    height = float(request.form['height'])
    weight =float(request.form['weight'])
    bmi =float(request.form['bmi'])

    print(f'Input Data: Age={age}, Gender={gender}, Height={height}, Weight={weight}, BMI={bmi}')


    # Perform prediction using the pre-trained model
    shape = np.array([age,gender,height,weight,bmi,]).reshape(1,-1)
    prediction = model.predict(shape)

    print(f'Prediction: {prediction}')


    
    return render_template('result.html', predicted_category=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)