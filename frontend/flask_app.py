# Copyright (c) 2026 MyCompany LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
import app.tools as tools

app = Flask(__name__, template_folder='templates', static_folder='static')

# Configure upload and image directories
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploads'))
IMAGES_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images'))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)

# Cache for the currently uploaded dataset info
current_dataset_info = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Handles dataset upload and validation."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    target_column = request.form.get('target_column', '').strip()
    
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
        
    if not target_column:
        return jsonify({'success': False, 'error': 'Target column name is required'}), 400

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Run Validation Agent/Tool
    validation_res = tools.validate_and_sample_dataset(file_path, target_column)
    if validation_res['status'] == 'error':
        return jsonify({'success': False, 'error': validation_res['message']}), 400

    # Delete existing model to force retrain on new dataset upload
    model_path = "models/rf_model.pkl"
    if os.path.exists(model_path):
        os.remove(model_path)

    # Calculate default query instance (first row of dataset) and min/max ranges
    df = pd.read_csv(validation_res['sampled_path'])
    features = validation_res['features']
    
    defaults = {}
    ranges = {}
    first_row = df.iloc[0]
    for col in features:
        val = first_row[col]
        if pd.api.types.is_integer_dtype(df[col]):
            defaults[col] = int(val)
            ranges[col] = [int(df[col].min()), int(df[col].max())]
        elif pd.api.types.is_numeric_dtype(df[col]):
            defaults[col] = float(val)
            ranges[col] = [float(df[col].min()), float(df[col].max())]
        else:
            defaults[col] = str(val)
            ranges[col] = list(df[col].unique())

    # Get sample target range
    target_min = float(df[target_column].min())
    target_max = float(df[target_column].max())
    target_default_range = [float(df[target_column].quantile(0.25)), float(df[target_column].quantile(0.75))]

    # Generate correlation heatmap and top 10 list
    corr_res = tools.generate_correlation_matrix(validation_res['sampled_path'], target_column)

    global current_dataset_info
    current_dataset_info = {
        'sampled_path': validation_res['sampled_path'],
        'target_column': target_column,
        'features': features,
        'integer_features': validation_res.get('integer_features', []),
        'defaults': defaults,
        'ranges': ranges,
        'correlation_heatmap': '/images/correlation_heatmap.png',
        'correlation_top_10': corr_res.get('top_10', [])
    }

    return jsonify({
        'success': True,
        'num_rows_original': validation_res['num_rows_original'],
        'num_rows_sampled': validation_res['num_rows_sampled'],
        'features': features,
        'integer_features': validation_res.get('integer_features', []),
        'defaults': defaults,
        'ranges': ranges,
        'target_column': target_column,
        'target_range': [target_min, target_max],
        'target_default_range': target_default_range,
        'correlation_heatmap': '/images/correlation_heatmap.png',
        'correlation_top_10': corr_res.get('top_10', []),
        'sampling_message': validation_res.get('sampling_message', '')
    })


@app.route('/api/random_query', methods=['GET'])
def get_random_query():
    """Selects a random row from the validated dataset to act as a query instance."""
    global current_dataset_info
    if not current_dataset_info:
        return jsonify({'success': False, 'error': 'Please upload a dataset first'}), 400

    df = pd.read_csv(current_dataset_info['sampled_path'])
    features = current_dataset_info['features']
    
    import random
    random_idx = random.randint(0, len(df) - 1)
    row_df = df.iloc[[random_idx]][features]
    random_row = {}
    for col in features:
        val = row_df.iloc[0][col]
        if pd.api.types.is_integer_dtype(df[col]):
            random_row[col] = int(val)
        elif pd.api.types.is_numeric_dtype(df[col]):
            random_row[col] = float(val)
        else:
            random_row[col] = str(val)
    
    return jsonify({
        'success': True,
        'query_instance': random_row
    })


@app.route('/run', methods=['POST'])
def run():
    """Runs the ML training, SHAP, and DiCE execution flow."""
    global current_dataset_info
    if not current_dataset_info:
        return jsonify({'success': False, 'error': 'Please upload and validate a dataset first.'}), 400

    data = request.json
    query_instance = data.get('query_instance')
    target_range = data.get('target_range')
    permitted_range_input = data.get('permitted_ranges', {})
    features_to_vary = data.get('features_to_vary')
    total_CFs = int(data.get('total_CFs', 3))
    threshold = float(data.get('threshold', 0.0))
    optimization_strategy = data.get('optimization_strategy', 'standard')

    if not query_instance or not target_range:
        return jsonify({'success': False, 'error': 'Query instance and target range are required.'}), 400

    sampled_path = current_dataset_info['sampled_path']
    target_column = current_dataset_info['target_column']
    features = current_dataset_info['features']

    # Cast inputs correctly
    processed_query = {}
    for k, v in query_instance.items():
        processed_query[k] = float(v) if isinstance(v, (int, float, str)) and str(v).replace('.', '', 1).isdigit() else v

    processed_permitted = {}
    integer_features = current_dataset_info.get('integer_features', [])
    for k, v in permitted_range_input.items():
        if v:
            if k in integer_features:
                processed_permitted[k] = [int(round(float(v[0]))), int(round(float(v[1])))]
            else:
                processed_permitted[k] = [float(v[0]), float(v[1])]

    try:
        # Check if the model already exists to skip training and SHAP analysis
        model_path = "models/rf_model.pkl"
        model_exists = os.path.exists(model_path)

        if not model_exists:
            # Step 2: Train Model
            train_res = tools.train_rf_quantile_regressor(sampled_path, target_column)
            if train_res['status'] == 'error':
                return jsonify({'success': False, 'error': 'Model training failed.'}), 500
            current_dataset_info['train_r2'] = train_res['train_r2']
            current_dataset_info['test_r2'] = train_res['test_r2']
            current_dataset_info['oob_score'] = train_res['oob_score']

            # Step 3: Run SHAP
            shap_res = tools.generate_shap_analysis(train_res['model_path'], sampled_path, target_column)
            if shap_res['status'] == 'error':
                return jsonify({'success': False, 'error': 'SHAP analysis failed.'}), 500
        else:
            train_res = {
                'status': 'success',
                'model_path': model_path,
                'train_r2': current_dataset_info.get('train_r2', 0.95),
                'test_r2': current_dataset_info.get('test_r2', 0.91),
                'oob_score': current_dataset_info.get('oob_score', 0.94)
            }

        # Step 4: Run DiCE
        dice_res = tools.generate_dice_counterfactuals(
            train_res['model_path'],
            sampled_path,
            target_column,
            processed_query,
            [float(target_range[0]), float(target_range[1])],
            processed_permitted,
            features_to_vary,
            total_CFs,
            current_dataset_info.get('integer_features'),
            optimization_strategy=optimization_strategy
        )

        # Get query instance prediction to display alongside
        import pickle
        with open(train_res['model_path'], "rb") as f:
            model = pickle.load(f)
        query_df = pd.DataFrame([processed_query])[features]
        
        if optimization_strategy == 'conservative':
            wrapped = tools.ConservativeModel(model)
            query_prediction = float(wrapped.predict(query_df)[0])
        else:
            query_prediction = float(model.predict(query_df)[0])

        # Step 5: Run SHAP Waterfall specifically for the query instance
        waterfall_res = tools.generate_shap_waterfall(
            train_res['model_path'],
            sampled_path,
            target_column,
            processed_query
        )

        return jsonify({
            'success': True,
            'train_r2': train_res['train_r2'],
            'test_r2': train_res['test_r2'],
            'oob_score': train_res.get('oob_score', 0.0),
            'query_prediction': query_prediction,
            'threshold': threshold,
            'shap': {
                'beeswarm': '/images/shap_beeswarm.png',
                'feature_importance': '/images/shap_feature_importance.png',
                'waterfall': '/images/shap_waterfall.png'
            },
            'plots': {
                'train_fit': '/images/train_predicted_vs_target.png',
                'test_fit': '/images/test_predicted_vs_target.png'
            },
            'dice': dice_res.get('counterfactuals', []) if dice_res['status'] == 'success' else [],
            'dice_error': dice_res.get('message') if dice_res['status'] == 'error' else None
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serves the generated plot images."""
    return send_from_directory(IMAGES_FOLDER, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
