"""
Model Utilities for Cloud Resource Forecasting
===============================================

This module provides utility functions for:
- Saving and loading trained models
- Calculating evaluation metrics
- Saving results in standardized format
- Multi-step forecasting with configurable horizon

Author: Cloud Resource Forecasting Project
"""

import json
import pickle
import os
from datetime import datetime
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================================================================
# MODEL PERSISTENCE
# ==============================================================================

def save_model(model, model_name: str, target: str, config: Dict[str, Any], 
               models_dir: str = 'models') -> str:
    """
    Save trained model with metadata
    
    Args:
        model: Trained model object
        model_name: Name of model type (e.g., 'arimax', 'lstm')
        target: Target variable name
        config: Model configuration dictionary
        models_dir: Directory to save models
        
    Returns:
        str: Path to saved model file
    """
    # Create models directory if not exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_name}_{target}_{timestamp}.pkl"
    filepath = os.path.join(models_dir, filename)
    
    # Package model with metadata
    model_package = {
        'model': model,
        'model_name': model_name,
        'target': target,
        'config': config,
        'timestamp': timestamp,
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save to file
    with open(filepath, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"✓ Model saved: {filepath}")
    return filepath


def load_model(filepath: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load trained model with metadata
    
    Args:
        filepath: Path to saved model file
        
    Returns:
        Tuple of (model, metadata_dict)
    """
    with open(filepath, 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    metadata = {
        'model_name': model_package['model_name'],
        'target': model_package['target'],
        'config': model_package['config'],
        'timestamp': model_package['timestamp'],
        'saved_at': model_package['saved_at']
    }
    
    print(f"✓ Model loaded: {filepath}")
    print(f"  Model: {metadata['model_name']}")
    print(f"  Target: {metadata['target']}")
    print(f"  Saved at: {metadata['saved_at']}")
    
    return model, metadata


def list_saved_models(models_dir: str = 'models', model_name: str = None, 
                      target: str = None) -> pd.DataFrame:
    """
    List all saved models with filtering options
    
    Args:
        models_dir: Directory containing models
        model_name: Filter by model name (optional)
        target: Filter by target variable (optional)
        
    Returns:
        DataFrame with model information
    """
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return pd.DataFrame()
    
    models_info = []
    
    for filename in os.listdir(models_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(models_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    package = pickle.load(f)
                
                info = {
                    'filename': filename,
                    'filepath': filepath,
                    'model_name': package.get('model_name', 'unknown'),
                    'target': package.get('target', 'unknown'),
                    'saved_at': package.get('saved_at', 'unknown'),
                    'config': str(package.get('config', {}))
                }
                
                # Apply filters
                if model_name and info['model_name'] != model_name:
                    continue
                if target and info['target'] != target:
                    continue
                
                models_info.append(info)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {str(e)}")
    
    return pd.DataFrame(models_info)


# ==============================================================================
# FORECASTING
# ==============================================================================

def rolling_forecast(model, y_test: pd.Series, X_test: pd.DataFrame, 
                    horizon: int = 20, model_type: str = 'arimax',
                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform rolling multi-step ahead forecasting
    
    Args:
        model: Trained model
        y_test: Test target values
        X_test: Test exogenous features
        horizon: Number of steps to forecast ahead
        model_type: Type of model ('arimax', 'lstm', etc.)
        verbose: Print progress
        
    Returns:
        Tuple of (predictions, actual_values)
    """
    n_test = len(y_test)
    predictions = []
    
    # Can only forecast where we have enough future data
    n_forecast_points = n_test - horizon + 1
    
    if verbose:
        print(f"  Forecasting {n_forecast_points} points with horizon={horizon}")
    
    for i in range(n_forecast_points):
        if model_type == 'arimax':
            # ARIMAX/SARIMAX specific forecasting
            exog_forecast = X_test.iloc[i:i+horizon]
            forecast = model.forecast(steps=horizon, exog=exog_forecast)
            predictions.append(forecast.iloc[-1])
        else:
            # Generic forecasting (adapt for other model types)
            raise NotImplementedError(f"Forecasting not implemented for {model_type}")
        
        if verbose and (i + 1) % 1000 == 0:
            print(f"    Progress: {i+1}/{n_forecast_points}")
    
    # Align actual values
    predictions = np.array(predictions)
    actual = y_test.iloc[horizon-1:horizon-1+n_forecast_points].values
    
    return predictions, actual


def forecast_with_horizon(model, y_test: pd.Series, X_test: pd.DataFrame,
                          horizon: int, model_type: str = 'arimax') -> Dict[str, Any]:
    """
    Perform forecasting and return results dictionary
    
    Args:
        model: Trained model
        y_test: Test target values  
        X_test: Test exogenous features
        horizon: Forecast horizon in steps
        model_type: Type of model
        
    Returns:
        Dictionary with predictions, actual values, and metadata
    """
    import time
    
    print(f"Forecasting with horizon={horizon} steps ({horizon*0.5:.1f} minutes)")
    
    start_time = time.time()
    predictions, actual = rolling_forecast(model, y_test, X_test, horizon, model_type)
    forecast_time = time.time() - start_time
    
    results = {
        'predictions': predictions,
        'actual': actual,
        'n_predictions': len(predictions),
        'forecast_time': forecast_time,
        'horizon': horizon,
        'horizon_minutes': horizon * 0.5
    }
    
    print(f"✓ Completed in {forecast_time:.2f}s")
    print(f"  Predictions: {len(predictions):,}")
    
    return results


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metrics: mae, rmse, mape, r2
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (handle division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    # Additional metrics
    max_error = np.max(np.abs(y_true - y_pred))
    median_ae = np.median(np.abs(y_true - y_pred))
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mse': float(mse),
        'mape': float(mape),
        'r2': float(r2),
        'max_error': float(max_error),
        'median_ae': float(median_ae)
    }


def print_metrics(metrics: Dict[str, float], target_name: str = ""):
    """
    Print metrics in formatted way
    
    Args:
        metrics: Dictionary of metrics
        target_name: Name of target variable
    """
    if target_name:
        print(f"\nMetrics for {target_name}:")
    else:
        print("\nMetrics:")
    
    print("=" * 60)
    print(f"  MAE:         {metrics['mae']:.6f}")
    print(f"  RMSE:        {metrics['rmse']:.6f}")
    print(f"  MAPE:        {metrics['mape']:.2f}%")
    print(f"  R²:          {metrics['r2']:.6f}")
    print(f"  Max Error:   {metrics['max_error']:.6f}")
    print(f"  Median AE:   {metrics['median_ae']:.6f}")
    print("=" * 60)


# ==============================================================================
# RESULTS PERSISTENCE
# ==============================================================================

def save_results(results: Dict[str, Any], filename: str = None) -> str:
    """
    Save evaluation results to JSON file
    
    Args:
        results: Dictionary containing all results
        filename: Output filename (auto-generated if None)
        
    Returns:
        str: Path to saved file
    """
    if filename is None:
        model_name = results.get('model', 'model')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results_{model_name}_{timestamp}.json"
    
    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results_clean = convert_types(results)
    
    with open(filename, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"✓ Results saved: {filename}")
    return filename


def load_results(filename: str) -> Dict[str, Any]:
    """
    Load results from JSON file
    
    Args:
        filename: Path to results file
        
    Returns:
        Dictionary with results
    """
    with open(filename, 'r') as f:
        results = json.load(f)
    
    print(f"✓ Results loaded: {filename}")
    return results


def compare_results(results_files: list) -> pd.DataFrame:
    """
    Compare results from multiple experiments
    
    Args:
        results_files: List of result JSON file paths
        
    Returns:
        DataFrame with comparison
    """
    comparison_data = []
    
    for filepath in results_files:
        results = load_results(filepath)
        
        for target, target_results in results.get('targets', {}).items():
            metrics = target_results.get('metrics', {})
            
            comparison_data.append({
                'file': os.path.basename(filepath),
                'model': results.get('model', 'unknown'),
                'target': target,
                'horizon': results.get('forecast_horizon', 'N/A'),
                'mae': metrics.get('mae', np.nan),
                'rmse': metrics.get('rmse', np.nan),
                'mape': metrics.get('mape', np.nan),
                'r2': metrics.get('r2', np.nan)
            })
    
    df = pd.DataFrame(comparison_data)
    return df


# ==============================================================================
# COMPLETE WORKFLOW FUNCTIONS
# ==============================================================================

def train_and_save_model(model_trainer_fn, model_name: str, target: str,
                         config: Dict[str, Any], models_dir: str = 'models') -> Tuple[Any, str]:
    """
    Train model and save it
    
    Args:
        model_trainer_fn: Function that trains and returns model
        model_name: Name of model type
        target: Target variable name
        config: Model configuration
        models_dir: Directory to save model
        
    Returns:
        Tuple of (trained_model, saved_filepath)
    """
    print(f"Training {model_name} for {target}...")
    model = model_trainer_fn()
    
    filepath = save_model(model, model_name, target, config, models_dir)
    return model, filepath


def evaluate_and_save(model, y_test: pd.Series, X_test: pd.DataFrame,
                     horizon: int, model_name: str, target: str,
                     config: Dict[str, Any], training_info: Dict[str, Any],
                     model_type: str = 'arimax') -> Dict[str, Any]:
    """
    Complete evaluation workflow: forecast, calculate metrics, save results
    
    Args:
        model: Trained model
        y_test: Test target values
        X_test: Test exogenous features
        horizon: Forecast horizon
        model_name: Name of model type
        target: Target variable
        config: Model configuration
        training_info: Training metadata
        model_type: Type of model for forecasting
        
    Returns:
        Dictionary with all results
    """
    # Forecast
    forecast_results = forecast_with_horizon(model, y_test, X_test, horizon, model_type)
    
    # Calculate metrics
    metrics = calculate_metrics(forecast_results['actual'], forecast_results['predictions'])
    
    # Print metrics
    print_metrics(metrics, target)
    
    # Compile results
    results = {
        'model': model_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'target': target,
        'forecast_horizon': horizon,
        'forecast_horizon_minutes': horizon * 0.5,
        'model_config': config,
        'training': training_info,
        'forecasting': {
            'n_predictions': forecast_results['n_predictions'],
            'time_seconds': forecast_results['forecast_time']
        },
        'metrics': metrics
    }
    
    return results


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def create_models_directory(models_dir: str = 'models'):
    """Create models directory if it doesn't exist"""
    os.makedirs(models_dir, exist_ok=True)
    print(f"✓ Models directory ready: {models_dir}")


def get_latest_model(models_dir: str = 'models', model_name: str = None,
                     target: str = None) -> str:
    """
    Get path to the most recently saved model
    
    Args:
        models_dir: Directory containing models
        model_name: Filter by model name
        target: Filter by target
        
    Returns:
        Path to latest model file
    """
    models_df = list_saved_models(models_dir, model_name, target)
    
    if len(models_df) == 0:
        raise FileNotFoundError("No models found matching criteria")
    
    # Sort by saved_at and get latest
    models_df = models_df.sort_values('saved_at', ascending=False)
    latest_path = models_df.iloc[0]['filepath']
    
    print(f"Latest model: {latest_path}")
    return latest_path


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("Model Utilities Module")
    print("=" * 80)
    print("\nExample Usage:")
    print("""
    # Save a model
    save_model(trained_model, 'arimax', 'memory_usage_pct', config)
    
    # Load a model
    model, metadata = load_model('models/arimax_memory_usage_pct_20240101.pkl')
    
    # Forecast with custom horizon
    results = forecast_with_horizon(model, y_test, X_test, horizon=40)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Save results
    save_results(results, 'results_arimax_custom.json')
    
    # List all models
    models_df = list_saved_models()
    print(models_df)
    
    # Compare results
    comparison = compare_results(['results1.json', 'results2.json'])
    print(comparison)
    """)


