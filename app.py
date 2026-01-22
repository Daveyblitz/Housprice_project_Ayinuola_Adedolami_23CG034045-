from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Path to the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'house_price_model.pkl')

# Load the model if it exists
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Please train the model first.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    
    if request.method == 'POST':
        if not model:
            return render_template('index.html', prediction_text="Error: Model not loaded. Please train the model first.")
        
        try:
            # Extract features from form
            overall_qual = int(request.form['overall_qual'])
            gr_liv_area = float(request.form['gr_liv_area'])
            garage_cars = int(request.form['garage_cars'])
            year_built = int(request.form['year_built'])
            total_bsmt_sf = float(request.form['total_bsmt_sf'])
            full_bath = int(request.form['full_bath'])
            
            # Prepare feature array (Order must match training: OverallQual, GrLivArea, GarageCars, YearBuilt, TotalBsmtSF, FullBath)
            features = np.array([[overall_qual, gr_liv_area, garage_cars, year_built, total_bsmt_sf, full_bath]])
            
            # Predict
            prediction = model.predict(features)
            output = round(prediction[0], 2)
            
            prediction_text = f"Estimated House Price: ${output:,.2f}"
            
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
