# ucidrugrepo_api/app/app.py

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import sys
import os
import json
from pathlib import Path
from werkzeug.utils import secure_filename

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from UCIDrugReview_model.predict import make_prediction

app = Flask(__name__)

# Custom JSON encoder that handles NumPy arrays and other non-serializable types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Set the custom encoder for Flask's jsonify
app.json_encoder = NumpyEncoder

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return get_upload_template()

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Handle GET requests
    if request.method == 'GET':
        return jsonify({
            'error': 'This endpoint requires a POST request with a CSV file',
            'instructions': 'Please use the form at the root URL to upload a CSV file'
        }), 405
    
    # Handle POST requests
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read the CSV file
            df = pd.read_csv(filepath)
            
            # Convert DataFrame to dictionary format expected by make_prediction
            data_in = {
                'uniqueID': df['uniqueID'].tolist() if 'uniqueID' in df.columns else [],
                'drugName': df['drugName'].tolist() if 'drugName' in df.columns else [],
                'condition': df['condition'].tolist() if 'condition' in df.columns else [],
                'review': df['review'].tolist() if 'review' in df.columns else [],
                'date': df['date'].tolist() if 'date' in df.columns else [],
                'usefulCount': df['usefulCount'].tolist() if 'usefulCount' in df.columns else []
            }
            
            # Save the data as JSON file
            json_filename = os.path.splitext(filename)[0] + '.json'
            json_filepath = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
            with open(json_filepath, 'w') as json_file:
                json.dump(data_in, json_file)
            
            # Make prediction
            result = make_prediction(input_data=data_in)
            
            # Convert any NumPy arrays in the result to Python lists
            if isinstance(result, np.ndarray):
                result = result.tolist()
            elif isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        result[key] = value.tolist()
            
            # Create a response dictionary
            response_data = {
                'result': result,
                'json_file_path': json_filepath
            }
            
            # Use our custom encoder when saving to JSON
            return app.response_class(
                response=json.dumps(response_data, cls=NumpyEncoder),
                status=200,
                mimetype='application/json'
            )
            
        except Exception as e:
            import traceback
            return jsonify({
                'error': str(e),
                'traceback': traceback.format_exc()
            }), 500
    
    return jsonify({'error': 'File type not allowed. Please upload a CSV file.'}), 400

def get_upload_template():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload CSV for Drug Review Prediction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            .container {
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #343a40;
            }
            .form-group {
                margin-bottom: 15px;
            }
            .btn {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
            }
            .btn:hover {
                background-color: #0069d9;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Upload CSV for Drug Review Prediction</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="file">Select CSV File:</label>
                    <input type="file" id="file" name="file" accept=".csv">
                </div>
                <div class="form-group">
                    <button type="submit" class="btn">Upload and Predict</button>
                </div>
            </form>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')