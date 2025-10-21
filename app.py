from flask import Flask, jsonify,request, render_template

import pickle

import numpy as np

model_path='best_model.pkl'  

with open(model_path, 'rb') as file:

    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')

def home():

    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():

    try:

        features = [float(x) for x in request.form.values()]

        final_features = [np.array(features)]

        prediction = model.predict(final_features)

        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text='Predicted Value: {}'.format(output))

    except Exception as e:

        return render_template('index.html', prediction_text='Error: {}'.format(str(e)))
    
    if __name__ == '__main__':
        app.run(debug=True)