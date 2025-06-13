from flask import Flask, request, jsonify, render_template, redirect, url_for, session, make_response
import pandas as pd
import numpy as np
import joblib
import json
import csv
import io
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = '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'

# Load machine learning models
model = joblib.load('./models/random_forest_regressor_model.pkl')
model_columns = joblib.load('./models/model_columnsCOPY.pkl')
encoder = joblib.load('./models/target_encoder.pkl')

# Data storage directories
DATA_DIR = 'data'
USERS_FILE = os.path.join(DATA_DIR, 'users.json')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')

# Ensure data directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

def load_users():
    """Load users from JSON file."""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def save_users(users_data):
    """Save users to JSON file."""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving users: {e}")
        return False

def get_user_predictions_file(user_email):
    """Generate file path for user's predictions."""
    safe_email = user_email.replace('@', '_at_').replace('.', '_dot_')
    return os.path.join(PREDICTIONS_DIR, f"{safe_email}_predictions.json")

def load_user_predictions(user_email):
    """Load predictions for a specific user."""
    predictions_file = get_user_predictions_file(user_email)
    if os.path.exists(predictions_file):
        try:
            with open(predictions_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []

def save_user_predictions(user_email, predictions):
    """Save predictions for a specific user."""
    predictions_file = get_user_predictions_file(user_email)
    try:
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving predictions: {e}")
        return False

def store_prediction_result(user_email, input_data, predicted_price):
    """Store prediction result in user's JSON file."""
    try:
        # Load existing predictions
        predictions = load_user_predictions(user_email)
        
        # Create new prediction entry
        prediction_data = {
            'user_email': user_email,
            'timestamp': datetime.now().isoformat(),
            'input_parameters': {
                'longitude': float(input_data['longitude']),
                'latitude': float(input_data['latitude']),
                'housing_median_age': float(input_data['housing_median_age']),
                'total_rooms': float(input_data['total_rooms']),
                'total_bedrooms': float(input_data['total_bedrooms']),
                'population': float(input_data['population']),
                'households': float(input_data['households']),
                'median_income': float(input_data['median_income']),
                'ocean_proximity': input_data['ocean_proximity']
            },
            'predicted_price': float(predicted_price),
            'prediction_id': f"{user_email}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Add to predictions list
        predictions.append(prediction_data)
        
        # Sort by timestamp (newest first)
        predictions.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Save updated predictions
        return save_user_predictions(user_email, predictions)
        
    except Exception as e:
        print(f"Error storing prediction: {e}")
        return False

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']
    
    users = load_users()
    
    if email in users and check_password_hash(users[email]['password'], password):
        session['user'] = email
        session['user_email'] = email
        return redirect(url_for('home'))
    else:
        return render_template('login.html', error="Invalid email or password")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        users = load_users()
        
        if email in users:
            return render_template('register.html', error="Email already registered")
        
        # Hash password and store user
        hashed_password = generate_password_hash(password)
        users[email] = {
            'password': hashed_password,
            'created_at': datetime.now().isoformat()
        }
        
        if save_users(users):
            return render_template('login.html', success="Registration successful! Please login.")
        else:
            return render_template('register.html', error="Registration failed. Please try again.")
    
    return render_template('register.html')

@app.route('/home')
def home():
    if 'user' in session:
        return render_template('home.html')
    else:
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('user_email', None)
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict_house_price():
    if 'user' not in session:
        return redirect(url_for('index'))

    if request.method == 'GET':
        return render_template('predict.html')

    try:
        # Parse input data from form
        input_data = request.form

        # Create a DataFrame with the user input
        user_input = pd.DataFrame({
            'longitude': [float(input_data['longitude'])],
            'latitude': [float(input_data['latitude'])],
            'housing_median_age': [float(input_data['housing_median_age'])],
            'total_rooms': [float(input_data['total_rooms'])],
            'total_bedrooms': [float(input_data['total_bedrooms'])],
            'population': [float(input_data['population'])],
            'households': [float(input_data['households'])],
            'median_income': [float(input_data['median_income'])],
            'ocean_proximity': [input_data['ocean_proximity']]
        })

        # Compute derived features
        user_input['rooms_per_household'] = user_input['total_rooms'] / user_input['households']
        user_input['bedrooms_per_room'] = user_input['total_bedrooms'] / user_input['total_rooms']
        user_input['population_per_household'] = user_input['population'] / user_input['households']
        user_input['log_median_income'] = np.log(user_input['median_income'])

        # Apply the encoder to the 'ocean_proximity' column
        user_input = encoder.transform(user_input)

        # Ensure the column names in user_input match those in the model's training data
        user_input = user_input.reindex(columns=model_columns, fill_value=0)

        # Make prediction
        predicted_price = model.predict(user_input)

        # Store prediction result in JSON file
        store_prediction_result(session.get('user_email'), input_data, predicted_price[0])

        # Return the predicted price
        prediction_text = f"The predicted house price is: ${predicted_price[0]:,.2f}"
        return render_template('predict.html', prediction_text=prediction_text)

    except ValueError as ve:
        return jsonify({'error': f"Invalid input: {ve}"}), 400
    except Exception as e:
        return jsonify({'error': f"An error occurred: {e}"}), 500

@app.route('/reports')
def reports():
    """Display historical prediction results for the authenticated user."""
    if 'user' not in session:
        return redirect(url_for('index'))
    
    try:
        user_email = session.get('user_email')
        predictions = load_user_predictions(user_email)
        
        # Format predictions for display
        formatted_predictions = []
        for prediction in predictions:
            formatted_predictions.append({
                'timestamp': prediction.get('timestamp'),
                'predicted_price': prediction.get('predicted_price'),
                'input_parameters': prediction.get('input_parameters'),
                'prediction_id': prediction.get('prediction_id')
            })
        
        return render_template('reports.html', predictions=formatted_predictions)
    
    except Exception as e:
        return f"An error occurred while retrieving reports: {e}"

@app.route('/data_analysis')
def data_analysis():
    """Display comprehensive data analysis dashboard."""
    if 'user' not in session:
        return redirect(url_for('index'))
    
    try:
        # Import here to avoid circular imports
        import sys
        import os
        
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        
        from complete_data_pipeline import RealEstateDataAnalyzer
        
        # Initialize analyzer
        data_analyzer = RealEstateDataAnalyzer('housing.csv')
        
        if data_analyzer.data is None:
            return render_template('data_analysis.html', 
                                 analysis=None, 
                                 error="Could not load housing dataset. Please ensure 'housing.csv' exists in the root directory.")
        
        # Generate comprehensive analysis
        dataset_overview = data_analyzer.get_dataset_overview()
        property_attributes = data_analyzer.analyze_property_attributes()
        selling_prices = data_analyzer.analyze_selling_prices()
        price_influences = data_analyzer.identify_key_price_influences()
        market_insights = data_analyzer.generate_market_insights()
        model_analysis = data_analyzer.create_predictive_model_analysis()
        
        analysis_data = {
            'dataset_overview': dataset_overview,
            'property_attributes': property_attributes,
            'selling_prices': selling_prices,
            'price_influences': price_influences,
            'market_insights': market_insights,
            'model_analysis': model_analysis
        }
        
        return render_template('data_analysis.html', analysis=analysis_data)
    
    except ImportError as e:
        return render_template('data_analysis.html', 
            analysis=None, 
            error=f"Could not import data analysis module: {e}")
    except Exception as e:
        return render_template('data_analysis.html', 
            analysis=None, 
            error=f"An error occurred while generating analysis: {e}")


@app.route('/download_report/<report_type>')
def download_report(report_type):
    """Generate downloadable reports in various formats."""
    if 'user' not in session:
        return redirect(url_for('index'))
    
    try:
        user_email = session.get('user_email')
        predictions = load_user_predictions(user_email)
        
        if report_type == 'csv':
            return generate_csv_report(predictions, user_email)
        elif report_type == 'json':
            return generate_json_report(predictions, user_email)
        elif report_type == 'txt':
            return generate_txt_report(predictions, user_email)
        else:
            return "Invalid report type specified", 400
            
    except Exception as e:
        return f"An error occurred while generating the report: {e}"

def generate_csv_report(predictions, user_email):
    """Generate a CSV report containing prediction history."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'Prediction ID', 'Timestamp', 'Longitude', 'Latitude', 
        'Housing Median Age', 'Total Rooms', 'Total Bedrooms',
        'Population', 'Households', 'Median Income', 'Ocean Proximity',
        'Predicted Price'
    ])
    
    # Write data rows
    for prediction in predictions:
        params = prediction.get('input_parameters', {})
        writer.writerow([
            prediction.get('prediction_id', ''),
            prediction.get('timestamp', ''),
            params.get('longitude', ''),
            params.get('latitude', ''),
            params.get('housing_median_age', ''),
            params.get('total_rooms', ''),
            params.get('total_bedrooms', ''),
            params.get('population', ''),
            params.get('households', ''),
            params.get('median_income', ''),
            params.get('ocean_proximity', ''),
            f"${prediction.get('predicted_price', 0):,.2f}"
        ])
    
    output.seek(0)
    
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = f'attachment; filename=prediction_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    return response

def generate_json_report(predictions, user_email):
    """Generate a JSON report containing prediction history."""
    report_data = {
        'user_email': user_email,
        'report_generated': datetime.now().isoformat(),
        'total_predictions': len(predictions),
        'predictions': predictions
    }
    
    response = make_response(json.dumps(report_data, indent=2))
    response.headers['Content-Type'] = 'application/json'
    response.headers['Content-Disposition'] = f'attachment; filename=prediction_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    return response

def generate_txt_report(predictions, user_email):
    """Generate a TXT report containing prediction history."""
    output = io.StringIO()
    
    # Write header
    output.write("REAL ESTATE PRICE PREDICTION REPORT\n")
    output.write("=" * 50 + "\n")
    output.write(f"User: {user_email}\n")
    output.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output.write(f"Total Predictions: {len(predictions)}\n")
    output.write("=" * 50 + "\n\n")
    
    # Write prediction details
    for i, prediction in enumerate(predictions, 1):
        params = prediction.get('input_parameters', {})
        output.write(f"PREDICTION #{i}\n")
        output.write("-" * 20 + "\n")
        output.write(f"ID: {prediction.get('prediction_id', 'N/A')}\n")
        output.write(f"Date: {prediction.get('timestamp', 'N/A')}\n")
        output.write(f"Predicted Price: ${prediction.get('predicted_price', 0):,.2f}\n")
        output.write(f"Location: {params.get('latitude', 'N/A')}, {params.get('longitude', 'N/A')}\n")
        output.write(f"Housing Age: {params.get('housing_median_age', 'N/A')} years\n")
        output.write(f"Total Rooms: {params.get('total_rooms', 'N/A')}\n")
        output.write(f"Total Bedrooms: {params.get('total_bedrooms', 'N/A')}\n")
        output.write(f"Population: {params.get('population', 'N/A')}\n")
        output.write(f"Households: {params.get('households', 'N/A')}\n")
        output.write(f"Median Income: ${params.get('median_income', 'N/A'):,.2f}\n")
        output.write(f"Ocean Proximity: {params.get('ocean_proximity', 'N/A')}\n")
        output.write("\n")
    
    output.seek(0)
    
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/plain'
    response.headers['Content-Disposition'] = f'attachment; filename=prediction_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    return response

@app.route('/clear', methods=['POST'])
def clear_prediction():
    return render_template('predict.html', prediction_text='')

if __name__ == '__main__':
    app.run(debug=True)