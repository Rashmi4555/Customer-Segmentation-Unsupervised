"""
Feature engineering module for customer segmentation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Creates customer-level features from transaction data"""
    
    def __init__(self, config=None):
        """
        Initialize FeatureEngineer with optional configuration
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config if config else {}
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_features(self, df):
        """Create customer-level features from transaction data"""
        print("  Creating customer features...")
        
        # Ensure we're working with a copy
        df = df.copy()
        
        # Identify key columns
        customer_col = self._find_column(df, ['CustomerID', 'CustomerId', 'Customer ID', 'Customer'])
        date_col = self._find_column(df, ['InvoiceDate', 'Date', 'TransactionDate'])
        invoice_col = self._find_column(df, ['InvoiceNo', 'Invoice', 'InvoiceNumber'])
        
        # Find value column
        value_col = None
        if 'TotalValue' in df.columns:
            value_col = 'TotalValue'
        elif 'Total' in df.columns:
            value_col = 'Total'
        elif 'Revenue' in df.columns:
            value_col = 'Revenue'
        
        if not value_col:
            if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
                df['TotalValue'] = df['Quantity'] * df['UnitPrice']
                value_col = 'TotalValue'
                print("  ✓ Created TotalValue from Quantity × UnitPrice")
            elif 'Quantity' in df.columns and 'Price' in df.columns:
                df['TotalValue'] = df['Quantity'] * df['Price']
                value_col = 'TotalValue'
                print("  ✓ Created TotalValue from Quantity × Price")
        
        if not customer_col:
            raise ValueError("No customer column found")
        
        if not value_col:
            raise ValueError("No value column found and cannot create TotalValue")
        
        # Set reference date
        if date_col and date_col in df.columns:
            snapshot_date = pd.to_datetime(df[date_col]).max() + timedelta(days=1)
        else:
            snapshot_date = datetime.now()
        
        # Calculate RFM
        if date_col and date_col in df.columns and invoice_col and invoice_col in df.columns:
            # Full RFM calculation
            rfm = df.groupby(customer_col).agg({
                date_col: lambda x: (snapshot_date - pd.to_datetime(x).max()).days,
                invoice_col: 'nunique',
                value_col: 'sum'
            }).reset_index()
            rfm.columns = [customer_col, 'Recency', 'Frequency', 'Monetary']
        elif date_col and date_col in df.columns:
            # No invoice column
            temp = df.groupby(customer_col).agg({
                date_col: lambda x: (snapshot_date - pd.to_datetime(x).max()).days,
                value_col: 'sum',
                customer_col: 'count'
            }).reset_index()
            temp.columns = [customer_col, 'Recency', 'Monetary', 'Frequency']
            rfm = temp[[customer_col, 'Recency', 'Frequency', 'Monetary']]
        else:
            # Minimal RFM
            temp = df.groupby(customer_col).agg({
                value_col: 'sum',
                customer_col: 'count'
            }).reset_index()
            temp.columns = [customer_col, 'Monetary', 'Frequency']
            temp['Recency'] = 0
            rfm = temp[[customer_col, 'Recency', 'Frequency', 'Monetary']]
        
        # Ensure all RFM columns are numeric
        rfm['Recency'] = pd.to_numeric(rfm['Recency'], errors='coerce').fillna(0)
        rfm['Frequency'] = pd.to_numeric(rfm['Frequency'], errors='coerce').fillna(0)
        rfm['Monetary'] = pd.to_numeric(rfm['Monetary'], errors='coerce').fillna(0)
        
        # Create RFM scores
        rfm = self._create_rfm_scores(rfm)
        
        # Add behavioral features
        customer_features = self._add_behavioral_features(df, rfm, customer_col, date_col, value_col)
        
        # Final check: ensure ALL features are numeric
        numeric_cols = customer_features.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric = [col for col in customer_features.columns if col not in numeric_cols and col != customer_col]
        
        if non_numeric:
            print(f"  ⚠️ Converting non-numeric columns: {non_numeric}")
            for col in non_numeric:
                customer_features[col] = pd.to_numeric(customer_features[col], errors='coerce').fillna(0)
        
        print(f"  ✓ Created {len(customer_features.columns)} features for {len(customer_features)} customers")
        print(f"  ✓ All features are now numeric")
        
        return customer_features
    
    def _find_column(self, df, possible_names):
        """Find a column from possible names"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def _create_rfm_scores(self, rfm):
        """Create RFM scores on 1-5 scale"""
        try:
            rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1]).astype(int)
        except:
            rfm['R_Score'] = pd.cut(rfm['Recency'].rank(pct=True),
                                     bins=[0,0.2,0.4,0.6,0.8,1.0],
                                     labels=[5,4,3,2,1]).astype(int)
        
        try:
            rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
        except:
            rfm['F_Score'] = pd.cut(rfm['Frequency'].rank(pct=True),
                                     bins=[0,0.2,0.4,0.6,0.8,1.0],
                                     labels=[1,2,3,4,5]).astype(int)
        
        try:
            rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
        except:
            rfm['M_Score'] = pd.cut(rfm['Monetary'].rank(pct=True),
                                     bins=[0,0.2,0.4,0.6,0.8,1.0],
                                     labels=[1,2,3,4,5]).astype(int)
        
        rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
        return rfm
    
    def _add_behavioral_features(self, df, rfm, customer_col, date_col, value_col):
        """Add behavioral features"""
        # Average order value
        avg_order = df.groupby(customer_col)[value_col].mean().reset_index()
        avg_order.columns = [customer_col, 'AvgOrderValue']
        
        # Merge
        features = rfm.merge(avg_order, on=customer_col, how='left')
        
        # Customer lifetime
        if date_col and date_col in df.columns:
            first_purchase = df.groupby(customer_col)[date_col].min().reset_index()
            first_purchase.columns = [customer_col, 'FirstPurchase']
            features = features.merge(first_purchase, on=customer_col, how='left')
            
            snapshot_date = pd.to_datetime(df[date_col]).max() + timedelta(days=1)
            features['CustomerLifetime'] = (pd.to_datetime(snapshot_date) - 
                                             pd.to_datetime(features['FirstPurchase'])).dt.days
            features['CustomerLifetime'] = features['CustomerLifetime'].clip(lower=1)
        else:
            features['CustomerLifetime'] = 365
        
        # Derived features - ensure numeric
        features['ValuePerDay'] = features['Monetary'] / features['CustomerLifetime']
        features['AvgTransactionValue'] = features['Monetary'] / features['Frequency'].clip(lower=1)
        features['PurchaseFrequency'] = features['Frequency'] / (features['CustomerLifetime'] / 30)
        
        # Product variety
        product_col = self._find_column(df, ['StockCode', 'ProductCode', 'Product'])
        if product_col:
            product_var = df.groupby(customer_col)[product_col].nunique().reset_index()
            product_var.columns = [customer_col, 'UniqueProducts']
            features = features.merge(product_var, on=customer_col, how='left')
        else:
            features['UniqueProducts'] = 0
        
        # Convert all to numeric and handle infinities
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            features[col] = features[col].replace([np.inf, -np.inf], 0)
            features[col] = features[col].fillna(0)
        
        # Convert any remaining object columns to numeric
        for col in features.columns:
            if col != customer_col and features[col].dtype == 'object':
                try:
                    features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
                except:
                    pass
        
        return features
    
    def scale_features(self, df):
        """Scale features for clustering"""
        print("  Scaling features...")
        
        # Select numeric features for clustering
        self.feature_names = [
            'Recency', 'Frequency', 'Monetary',
            'R_Score', 'F_Score', 'M_Score', 'RFM_Score',
            'AvgOrderValue', 'ValuePerDay', 'AvgTransactionValue',
            'CustomerLifetime', 'PurchaseFrequency', 'UniqueProducts'
        ]
        
        # Keep only columns that exist
        self.feature_names = [col for col in self.feature_names if col in df.columns]
        
        # Extract features and ensure numeric
        X = df[self.feature_names].copy()
        
        # Double-check all are numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Convert to numpy array with float dtype
        X_array = X.values.astype(np.float64)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X_array)
        
        print(f"  ✓ Scaled {len(self.feature_names)} features")
        print(f"  ✓ Feature matrix shape: {X_scaled.shape}")
        
        return X_scaled, self.scaler
    
    def apply_pca(self, X_scaled, n_components=0.95):
        """Apply PCA for dimensionality reduction"""
        print("  Applying PCA...")
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"  ✓ Reduced to {X_pca.shape[1]} components")
        print(f"  ✓ Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return X_pca, pca