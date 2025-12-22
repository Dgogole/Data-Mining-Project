"""
Baseline regression models using scikit-learn on Boston Housing dataset
Compares performance of various sklearn regression algorithms
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from dcn.dcn import DCN
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
import torch
from time import sleep
warnings.filterwarnings('ignore')

from rrl.utils import read_csv, DBEncoder

DATA_DIR = '../data/boston_housing/'

def evaluate(model, dataloader):
    y_pred = []
    for batch_data, labels in dataloader:
        model.eval()
        with torch.no_grad():
            preds = model(batch_data)
        y_pred.append(preds)
    return torch.cat(y_pred, axis=0)

def train_model(model, X_train, y_train, X_test, y_test, model_name):
    """Train and evaluate a regression model"""
    # Train the model
    
    train_set = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.float32)).view(-1, 1))
    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test.astype(np.float32)).view(-1, 1))
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=128, shuffle=False)

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_func = MSELoss()

    for epoch in range(0, 500):
        step = 0
        tot_loss = 0
        for batch_data, labels in train_dataloader:
            model.train()
            with torch.enable_grad():
                preds = model(batch_data)
                loss = loss_func(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            step += 1
            tot_loss += loss
        if epoch % 20 == 19:
            print(f"Epoch {epoch + 1}: tot_loss: {tot_loss / step}")

    # Make predictions
    y_train_pred = evaluate(model, train_dataloader)
    y_test_pred = evaluate(model, test_dataloader)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    print({
        'model': model_name,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2
    })
    
    return {
        'model': model_name,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2
    }


def main():
    print("=" * 80)
    print("DCN on Boston Housing Dataset")
    print("=" * 80)
    
    # Load data
    dataset = 'boston_housing'
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    
    print(f"\nLoading data from {data_path}")
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)
    print(f"Loaded {X_df.shape[0]} samples with {X_df.shape[1]} features")
    
    # Encode data (similar to RRL preprocessing)
    db_enc = DBEncoder(f_df, discrete=False, y_one_hot=False, drop=None)
    db_enc.fit(X_df, y_df, task_type='regression')
    discrete_catenum = [len(x) for x in db_enc.feature_enc.categories_]
    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True, use_log=True)
    print(f"After encoding: X shape = {X.shape}, y shape = {y.shape}")
    
    results = []
    # Split data using 5-fold CV (same as RRL)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, test_index in list(kf.split(X)):
    
        X_train = X[train_index]
        y_train = y[train_index].ravel() if y.ndim > 1 else y[train_index]
        X_test = X[test_index]
        y_test = y[test_index].ravel() if y.ndim > 1 else y[test_index]

        
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)
        
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
                
        print("\n" + "=" * 80)
        print("Training and Evaluating Models")
        print("=" * 80)
        
        model = DCN(discrete_catenum, db_enc.continuous_flen, 128, 2, 2, "regression")
        result = train_model(model, X_train, y_train, X_test, y_test, "DCN")
            
        print(f"  Train MSE: {result['train_mse']:.4f}, Test MSE: {result['test_mse']:.4f}")
        print(f"  Train MAE: {result['train_mae']:.4f}, Test MAE: {result['test_mae']:.4f}")
        print(f"  Train R²:  {result['train_r2']:.4f}, Test R²:  {result['test_r2']:.4f}")
        results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_mse')
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary: Models sorted by Test MSE (lower is better)")
    print("=" * 80)
    print(results_df[['model', 'test_mse', 'test_mae', 'test_r2']].to_string(index=False))
    
    # Save results
    output_file = os.path.join(DATA_DIR, 'baseline_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print best model
    best_model = results_df.iloc[0]
    print("\n" + "=" * 80)
    print("Best Model (by Test MSE):")
    print("=" * 80)
    print(f"Model: {best_model['model']}")
    print(f"Test MSE: {best_model['test_mse']:.4f}")
    print(f"Test MAE: {best_model['test_mae']:.4f}")
    print(f"Test R²:  {best_model['test_r2']:.4f}")
    
    # Cross-validation on best model
    print("\n" + "=" * 80)
    print("5-Fold Cross-Validation on Best Model")
    print("=" * 80)
    best_model_name = best_model['model']
    best_model_instance = model
    
    if 'SVR' in best_model_name or 'MLP' in best_model_name:
        X_all_scaled = scaler.fit_transform(X)
        cv_scores = cross_val_score(best_model_instance, X_all_scaled, y.ravel() if y.ndim > 1 else y, 
                                   cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    else:
        cv_scores = cross_val_score(best_model_instance, X, y.ravel() if y.ndim > 1 else y, 
                                   cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    
    cv_mse = -cv_scores
    print(f"CV MSE: {cv_mse.mean():.4f} (+/- {cv_mse.std() * 2:.4f})")
    print(f"CV MSE per fold: {cv_mse}")
    
    print("\n" + "=" * 80)
    print("Baseline Evaluation Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

    