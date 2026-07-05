import os
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats.qmc import LatinHypercube
import shap
import dice_ml
import pickle

# Ensure images directory exists
os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)


def validate_and_sample_dataset(file_path: str, target_column: str) -> dict:
    """Validates the input dataset and performs the required sampling logic.

    Args:
        file_path: Path to the CSV or Excel file.
        target_column: Name of the target variable/column.

    Returns:
        A dictionary containing the status, validation details, and paths to saved sampled data.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        return {"status": "error", "message": "Unsupported file format. Please upload a .csv or .xlsx file."}

    # Identify features (excluding target)
    features = [col for col in df.columns if col != target_column]
    if len(features) > 10:
        return {"status": "error", "message": f"Dataset has {len(features)} features, which exceeds the maximum limit of 10."}

    if target_column not in df.columns:
        return {"status": "error", "message": f"Target column '{target_column}' not found in the dataset."}

    # Drop missing values in critical columns
    df = df.dropna(subset=[target_column] + features)
    num_rows = len(df)
    sampled_df = df.copy()

    # Sampling Logic
    if num_rows >= 3000:
        if 5000 <= num_rows <= 10000:
            # Randomly select 3000 samples
            sampled_df = df.sample(n=3000, random_state=42)
        elif num_rows > 10000:
            # Latin Hypercube Sampling (LHS) to select 3000 samples
            num_features = len(features)
            # LHS generator
            sampler = LatinHypercube(d=num_features, seed=42)
            lhs_samples = sampler.random(n=3000)
            
            # Extract feature min-max ranges
            feature_min = df[features].min().values
            feature_max = df[features].max().values
            
            # Avoid division by zero if min == max
            feature_range = feature_max - feature_min
            feature_range[feature_range == 0] = 1.0
            
            # Map LHS samples to the feature space ranges
            lhs_mapped = feature_min + lhs_samples * feature_range
            
            # Find closest actual rows in df to the LHS mapped points (using normalized distance)
            df_norm = (df[features] - feature_min) / feature_range
            lhs_norm = lhs_samples # LHS is already in [0, 1]^d
            
            # Efficient nearest neighbor lookup
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn.fit(df_norm.values)
            indices = nn.kneighbors(lhs_norm, return_distance=False).flatten()
            
            # Select unique indices or keep mapping (LHS might map to same closest point, but we want 3000 unique if possible,
            # if not we take the mapped indices)
            unique_indices = np.unique(indices)
            if len(unique_indices) < 3000:
                # If we got duplicates, fill up the rest using random sampling
                remaining_needed = 3000 - len(unique_indices)
                all_idx = np.arange(len(df))
                leftover_idx = np.setdiff1d(all_idx, unique_indices)
                fill_idx = np.random.choice(leftover_idx, size=remaining_needed, replace=False)
                final_indices = np.concatenate([unique_indices, fill_idx])
            else:
                final_indices = unique_indices[:3000]
                
            sampled_df = df.iloc[final_indices]
        else:
            # Fallback for 3000 <= num_rows < 5000
            sampled_df = df.sample(n=3000, random_state=42)

    # Save sampled dataset
    sampled_path = "sampled_dataset.csv"
    sampled_df.to_csv(sampled_path, index=False)

    # Identify integer features
    integer_features = [col for col in features if pd.api.types.is_integer_dtype(df[col])]

    return {
        "status": "success",
        "num_rows_original": num_rows,
        "num_rows_sampled": len(sampled_df),
        "features": features,
        "integer_features": integer_features,
        "target": target_column,
        "sampled_path": sampled_path
    }


def train_rf_quantile_regressor(sampled_path: str, target_column: str) -> dict:
    """Trains a Random Forest Regressor and extracts 95% prediction intervals.
    Generates Predicted vs Target plots for train and test sets.
    """
    df = pd.read_csv(sampled_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    # Save the trained model
    model_path = "models/rf_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)

    # Predict train and test target values
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    # Calculate 95% prediction intervals on the training set (2.5% and 97.5% quantiles from estimators)
    # Collect predictions from all estimators: shape (n_samples, n_estimators)
    train_predictions = np.array([dt.predict(X_train) for dt in rf.estimators_]).T
    lower_bound = np.percentile(train_predictions, 2.5, axis=1)
    upper_bound = np.percentile(train_predictions, 97.5, axis=1)

    # Create combined 'Predicted vs Target' plot overlaying Train (with 95% PI) and Test set
    plt.figure(figsize=(10, 6))
    
    # 1. Train predictions error bounds
    y_err = [y_train_pred - lower_bound, upper_bound - y_train_pred]
    y_err[0] = np.maximum(y_err[0], 0)
    y_err[1] = np.maximum(y_err[1], 0)
    
    # Plot Train with error bars
    plt.errorbar(y_train, y_train_pred, yerr=y_err, fmt='o', color='blue', alpha=0.3, 
                 label='Train Predictions (with 95% PI)', elinewidth=0.6, capsize=1)
    
    # 2. Plot Test predictions as dark red dots
    plt.scatter(y_test, y_test_pred, color='darkred', alpha=0.75, edgecolors='black', 
                label='Test Predictions', zorder=5)

    # Plot diagonal reference line
    all_y = np.concatenate([y_train, y_test])
    plt.plot([all_y.min(), all_y.max()], [all_y.min(), all_y.max()], 'r--', lw=2, label='Perfect Fit')
    
    plt.xlabel('True Target Value')
    plt.ylabel('Predicted Target Value')
    plt.title('Predicted vs Target Overlay (Train & Test with Train 95% PI)')
    plt.legend()
    plt.tight_layout()
    
    # Save training plot
    train_plot_path = "images/train_predicted_vs_target.png"
    plt.savefig(train_plot_path)
    # Also save as test path to prevent breaking existing code
    test_plot_path = "images/test_predicted_vs_target.png"
    plt.savefig(test_plot_path)
    plt.close()

    # Also compute overall metrics
    from sklearn.metrics import mean_squared_error, r2_score
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return {
        "status": "success",
        "train_r2": r2_train,
        "test_r2": r2_test,
        "oob_score": float(rf.oob_score_),
        "model_path": model_path,
        "train_plot_path": train_plot_path,
        "test_plot_path": test_plot_path
    }


def generate_shap_analysis(model_path: str, sampled_path: str, target_column: str) -> dict:
    """Uses TreeExplainer to generate Shapley values, beeswarm, and feature importance plots."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(sampled_path)
    X = df.drop(columns=[target_column])

    # TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # Generate Beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    beeswarm_path = "images/shap_beeswarm.png"
    plt.savefig(beeswarm_path)
    plt.close()

    # Generate Feature Importance plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    bar_path = "images/shap_feature_importance.png"
    plt.savefig(bar_path)
    plt.close()

    return {
        "status": "success",
        "beeswarm_path": beeswarm_path,
        "feature_importance_path": bar_path
    }


def generate_dice_counterfactuals(model_path: str, sampled_path: str, target_column: str, query_instance_dict: dict, target_range: list, permitted_ranges: dict = None, features_to_vary: list = None, total_CFs: int = 3, integer_features: list = None) -> dict:
    """Generates DiCE counterfactuals to show how features need to change to reach a target range."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(sampled_path)
    features = [col for col in df.columns if col != target_column]

    # Construct DiCE data and model objects
    d = dice_ml.Data(dataframe=df, continuous_features=features, outcome_name=target_column)
    m = dice_ml.Model(model=model, backend="sklearn", model_type="regressor")
    
    # Initialize DiCE explainer
    exp = dice_ml.Dice(d, m, method="random")

    # Format query instance as DataFrame and ensure correct column ordering
    query_instance = pd.DataFrame([query_instance_dict])[features]

    if features_to_vary is None:
        features_to_vary = "all"

    # Filter permitted ranges to only include features we want to vary
    if permitted_ranges and features_to_vary != "all":
        permitted_ranges = {k: v for k, v in permitted_ranges.items() if k in features_to_vary}

    try:
        # Generate counterfactuals
        cf = exp.generate_counterfactuals(
            query_instance,
            total_CFs=total_CFs,
            desired_range=target_range,
            permitted_range=permitted_ranges,
            features_to_vary=features_to_vary
        )
        
        # Convert to dictionary/dataframe format to return
        cf_df = cf.cf_examples_list[0].final_cfs_df
        if cf_df is not None:
            cfs_list = cf_df.to_dict(orient="records")
            # Round integer features
            if integer_features:
                for row in cfs_list:
                    for feat in integer_features:
                        if feat in row:
                            row[feat] = int(round(row[feat]))
            return {
                "status": "success",
                "counterfactuals": cfs_list
            }
        else:
            return {"status": "error", "message": "Failed to find valid counterfactuals."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def generate_correlation_matrix(sampled_path: str, target_column: str) -> dict:
    """Generates a correlation heatmap and extracts the top 10 largest correlation contributions."""
    df = pd.read_csv(sampled_path)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    # Generate heatmap
    plt.figure(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, annot_kws={"size": 9})
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    heatmap_path = "images/correlation_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()

    # Extract top 10 unique contributions (excluding self-correlations)
    corr_pairs = corr.unstack()
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    
    unique_pairs = []
    seen = set()
    for (f1, f2), val in corr_pairs.items():
        pair_key = tuple(sorted([f1, f2]))
        if pair_key not in seen:
            seen.add(pair_key)
            unique_pairs.append({
                "feature1": f1,
                "feature2": f2,
                "correlation": float(val),
                "abs_correlation": abs(float(val))
            })
            
    unique_pairs = sorted(unique_pairs, key=lambda x: x["abs_correlation"], reverse=True)
    top_10 = unique_pairs[:10]

    return {
        "status": "success",
        "heatmap_path": heatmap_path,
        "top_10": top_10
    }


def generate_shap_waterfall(model_path: str, sampled_path: str, target_column: str, query_instance_dict: dict) -> dict:
    """Generates a SHAP waterfall plot for the current query instance."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(sampled_path)
    features = [col for col in df.columns if col != target_column]

    processed_query = {}
    for k, v in query_instance_dict.items():
        processed_query[k] = float(v) if isinstance(v, (int, float, str)) and str(v).replace('.', '', 1).isdigit() else v

    query_df = pd.DataFrame([processed_query])[features]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(query_df)

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    waterfall_path = "images/shap_waterfall.png"
    plt.savefig(waterfall_path)
    plt.close()

    return {
        "status": "success",
        "waterfall_path": waterfall_path
    }
