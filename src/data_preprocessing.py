"""
Data preprocessing module for customer segmentation
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles all data preprocessing steps"""
    
    def __init__(self, config=None):
        """
        Initialize DataPreprocessor with optional configuration
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config if config else {}
        self.report_data = {}
    
    def load_data(self, filepath):
        """Load dataset from CSV file"""
        print(f"  Loading data from {filepath}")
        try:
            # Check if file exists
            if not Path(filepath).exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            df = pd.read_csv(filepath, encoding='latin1')
            print(f"  ✓ Loaded {len(df):,} records")
            return df
        except Exception as e:
            print(f"  ✗ Error loading data: {e}")
            raise
    
    def process_data(self, df):
        """Main preprocessing pipeline"""
        print("  Processing data...")
        original_shape = df.shape
        
        # Make a copy
        df_processed = df.copy()
        
        # Standardize column names (remove spaces)
        df_processed.columns = df_processed.columns.str.strip()
        
        # Find customer column
        customer_col = None
        possible_customer = ['CustomerID', 'CustomerId', 'Customer ID', 'Customer', 'Customer_No']
        for col in possible_customer:
            if col in df_processed.columns:
                customer_col = col
                break
        
        if customer_col is None:
            print("  ⚠️ No customer column found! Using first column as ID")
            customer_col = df_processed.columns[0]
        
        # 1. Handle missing values
        missing_before = df_processed[customer_col].isnull().sum()
        df_processed = df_processed.dropna(subset=[customer_col])
        print(f"  ✓ Removed {missing_before:,} records with missing customer ID")
        
        # 2. Find invoice column
        invoice_col = None
        possible_invoice = ['InvoiceNo', 'Invoice', 'InvoiceNumber', 'Invoice_No']
        for col in possible_invoice:
            if col in df_processed.columns:
                invoice_col = col
                break
        
        # Remove cancelled transactions if invoice column exists
        if invoice_col:
            cancelled_mask = df_processed[invoice_col].astype(str).str.startswith('C')
            cancelled_count = cancelled_mask.sum()
            df_processed = df_processed[~cancelled_mask]
            print(f"  ✓ Removed {cancelled_count:,} cancelled transactions")
        
        # 3. Remove invalid quantities
        if 'Quantity' in df_processed.columns:
            invalid_qty = (df_processed['Quantity'] <= 0).sum()
            df_processed = df_processed[df_processed['Quantity'] > 0]
            print(f"  ✓ Removed {invalid_qty:,} invalid quantities")
        
        # 4. Remove invalid prices
        price_col = None
        if 'UnitPrice' in df_processed.columns:
            price_col = 'UnitPrice'
        elif 'Price' in df_processed.columns:
            price_col = 'Price'
        
        if price_col:
            invalid_price = (df_processed[price_col] <= 0).sum()
            df_processed = df_processed[df_processed[price_col] > 0]
            print(f"  ✓ Removed {invalid_price:,} invalid prices")
        
        # 5. Convert date column
        date_col = None
        if 'InvoiceDate' in df_processed.columns:
            date_col = 'InvoiceDate'
        elif 'Date' in df_processed.columns:
            date_col = 'Date'
        
        if date_col:
            df_processed[date_col] = pd.to_datetime(df_processed[date_col])
        
        # 6. Create TotalValue column if needed
        if 'TotalValue' not in df_processed.columns:
            if 'Quantity' in df_processed.columns and price_col:
                df_processed['TotalValue'] = df_processed['Quantity'] * df_processed[price_col]
                print(f"  ✓ Created TotalValue column")
        
        final_shape = df_processed.shape
        removed = original_shape[0] - final_shape[0]
        removed_pct = (removed / original_shape[0]) * 100 if original_shape[0] > 0 else 0
        
        print(f"\n  ✓ Preprocessing complete:")
        print(f"    Original: {original_shape[0]:,} records")
        print(f"    Final: {final_shape[0]:,} records")
        print(f"    Removed: {removed:,} records ({removed_pct:.1f}%)")
        
        return df_processed
    
    def get_summary(self):
        """Get preprocessing summary"""
        return self.report_data