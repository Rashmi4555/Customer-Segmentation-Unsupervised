"""
Utility functions for the customer segmentation project
"""

import yaml
import json
import pickle
import logging
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration with automatic directory creation"""
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename
    log_filename = log_dir / f'customer_segmentation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Setup logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    print(f"✅ Logging initialized. Log file: {log_filename}")

def ensure_directories():
    """Create all required directories for the project"""
    directories = [
        'data/raw',
        'data/processed',
        'logs',
        'results',
        'results/cluster_plots',
        'results/pca_outputs',
        'results/metrics',
        'reports',
        'models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ All directories created/verified")
    return True

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file with error handling"""
    config_path = Path(config_path)
    
    # Check if file exists
    if not config_path.exists():
        print(f"⚠️ Config file {config_path} not found. Using default configuration.")
        
        # Return default configuration
        return {
            'data': {
                'raw': 'data/raw/online_retail_II.csv',
                'processed': 'data/processed/',
                'cleaned': 'data/processed/cleaned_data.csv',
                'features': 'data/processed/customer_features.csv'
            },
            'features': {
                'rfm': ['recency', 'frequency', 'monetary'],
                'behavioral': ['avg_transaction_value', 'unique_products', 'purchase_frequency'],
                'derived': ['value_per_transaction', 'quantity_per_transaction']
            },
            'clustering': {
                'kmeans': {'max_clusters': 10, 'random_state': 42},
                'hierarchical': {'max_clusters': 10, 'linkage': 'ward'},
                'dbscan': {'eps': 0.5, 'min_samples': 5},
                'gmm': {'max_components': 10, 'covariance_type': 'full'}
            },
            'pca': {'n_components': 0.95},
            'output': {
                'results': 'results/',
                'plots': 'results/cluster_plots/',
                'pca': 'results/pca_outputs/',
                'metrics': 'results/metrics/',
                'logs': 'logs/'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    # Load existing config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded configuration from {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}. Using defaults.")
        return {}

def save_config(config: Dict, config_path: str):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def save_model(model: Any, model_path: str):
    """Save trained model to file"""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(model_path: str) -> Any:
    """Load trained model from file"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def save_dataframe(df: pd.DataFrame, filepath: str, **kwargs):
    """Save DataFrame to various formats"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.csv':
        df.to_csv(path, **kwargs)
    elif path.suffix == '.parquet':
        df.to_parquet(path, **kwargs)
    elif path.suffix == '.pkl':
        df.to_pickle(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

def load_dataframe(filepath: str, **kwargs) -> pd.DataFrame:
    """Load DataFrame from various formats"""
    path = Path(filepath)
    
    if path.suffix == '.csv':
        return pd.read_csv(path, **kwargs)
    elif path.suffix == '.parquet':
        return pd.read_parquet(path, **kwargs)
    elif path.suffix == '.pkl':
        return pd.read_pickle(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

def create_directory_structure(base_path: str = '.'):
    """Create project directory structure"""
    directories = [
        'data/raw',
        'data/processed',
        'notebooks',
        'src/clustering',
        'reports',
        'results/cluster_plots',
        'results/pca_outputs',
        'results/metrics',
        'logs',
        'models'
    ]
    
    for directory in directories:
        Path(base_path, directory).mkdir(parents=True, exist_ok=True)
    
    print(f"Directory structure created at {base_path}")

def calculate_statistics(df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """Calculate statistics for each cluster"""
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    # Group statistics
    cluster_stats = df_with_clusters.groupby('Cluster').agg(['mean', 'std', 'count'])
    
    # Flatten multi-index columns
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
    
    return cluster_stats

def detect_anomalies(df: pd.DataFrame, method: str = 'isolation_forest', **kwargs):
    """Detect anomalies in data"""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    
    if method == 'isolation_forest':
        clf = IsolationForest(**kwargs)
        anomalies = clf.fit_predict(df)
        return anomalies
    
    elif method == 'lof':
        lof = LocalOutlierFactor(**kwargs)
        anomalies = lof.fit_predict(df)
        return anomalies
    
    else:
        raise ValueError(f"Unknown anomaly detection method: {method}")

def calculate_feature_importance(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Calculate feature importance for clustering"""
    from sklearn.ensemble import RandomForestClassifier
    
    # Use RandomForest to determine feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, labels)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df

def generate_report(cluster_results: Dict, algorithm_name: str, output_dir: str = 'reports'):
    """Generate a comprehensive report for clustering results"""
    report = {
        'algorithm': algorithm_name,
        'timestamp': datetime.now().isoformat(),
        'cluster_summary': {},
        'metrics': {},
        'business_insights': {}
    }
    
    # Add cluster information
    for key, value in cluster_results.items():
        if isinstance(value, (int, float, str, bool, list, dict)):
            report['cluster_summary'][key] = value
    
    # Save report
    output_path = Path(output_dir) / f'{algorithm_name}_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4, default=str)
    
    return output_path

def validate_data(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """Validate input data"""
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if df.isnull().all().any():
        raise ValueError("Some columns contain only null values")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def memory_usage(df: pd.DataFrame) -> str:
    """Calculate memory usage of DataFrame"""
    memory = df.memory_usage(deep=True).sum()
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if memory < 1024:
            return f"{memory:.2f} {unit}"
        memory /= 1024
    
    return f"{memory:.2f} TB" 