 
import pandas as pd
import numpy as np
import os

def create_directories():
    """Create necessary directories"""
    os.makedirs('data/reference', exist_ok=True)
    os.makedirs('data/incoming', exist_ok=True)
    print("✅ Directories created")

def generate_training_data():
    """Generate training data"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 10, n_samples).astype(int),
        'income': np.random.normal(50000, 15000, n_samples).astype(int),
        'transaction_amount': np.random.exponential(100, n_samples),
        'credit_score': np.random.normal(700, 50, n_samples).astype(int),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'occupation': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Artist'], n_samples),
        'location': np.random.choice(['NY', 'CA', 'TX', 'FL'], n_samples),
        'product_category': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    df['age'] = np.clip(df['age'], 18, 80)
    df['income'] = np.clip(df['income'], 20000, 200000)
    df['credit_score'] = np.clip(df['credit_score'], 300, 850)
    return df

def generate_drift_data():
    """Generate data with drift"""
    np.random.seed(123)
    n_samples = 500
    
    data = {
        'age': np.random.normal(45, 12, n_samples).astype(int),
        'income': np.random.normal(65000, 20000, n_samples).astype(int),
        'transaction_amount': np.random.exponential(150, n_samples),
        'credit_score': np.random.normal(680, 55, n_samples).astype(int),
        'gender': np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4]),
        'occupation': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Artist'], n_samples, p=[0.4, 0.2, 0.2, 0.2]),
        'location': np.random.choice(['NY', 'CA', 'TX', 'FL'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'product_category': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.2, 0.5, 0.3]),
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    df = pd.DataFrame(data)
    df['age'] = np.clip(df['age'], 18, 80)
    df['income'] = np.clip(df['income'], 20000, 200000)
    df['credit_score'] = np.clip(df['credit_score'], 300, 850)
    return df

if __name__ == "__main__":
    print("="*50)
    print("DATA GENERATION")
    print("="*50)
    
    create_directories()
    
    # Generate training data
    train_data = generate_training_data()
    train_data.to_csv('data/reference/training_data.csv', index=False)
    print(f"✅ Training data: {len(train_data)} samples")
    
    # Generate drift data
    drift_data = generate_drift_data()
    drift_data.to_csv('data/incoming/drift_sample.csv', index=False)
    print(f"✅ Drift data: {len(drift_data)} samples")
    
    print("\n✅ Data generation complete!")