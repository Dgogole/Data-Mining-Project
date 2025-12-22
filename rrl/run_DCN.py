import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score
import warnings
from dcn.dcn import DCN
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import MSELoss, BCELoss
from torch.optim import Adam
import torch
from copy import deepcopy
from time import sleep
warnings.filterwarnings('ignore')

from rrl.utils import read_csv, DBEncoder

TASK = "classification"
if TASK == "regression":
    DATA_DIR = '../data/boston_housing/'
elif TASK == 'classification':
    DATA_DIR = '../data/bank_marketing/'

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model, dataloader, task_type = "regression"):
    y_pred = []
    for batch_data, labels in dataloader:
        model.eval()
        batch_data = batch_data.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            preds = model(batch_data)
        if task_type == "classification":
            preds = (preds >= 0.5).long()
        y_pred.append(preds)
    return torch.cat(y_pred, axis=0).cpu()

def train_model(model, X_train, y_train, X_test, y_test, model_name, task_type = "regression"):
    """Train and evaluate a regression model"""
    # Train the model
    
    valid_ratio = 0.95
    sp = int(X_train.shape[0] * valid_ratio)
    X_train, X_valid = X_train[:sp], X_train[sp:]
    y_train, y_valid = y_train[:sp], y_train[sp:]
    train_set = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.float32)).view(-1, 1))
    valid_set = TensorDataset(torch.tensor(X_valid.astype(np.float32)), torch.tensor(y_valid.astype(np.float32)).view(-1, 1))
    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test.astype(np.float32)).view(-1, 1))
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=False)
    valid_dataloader = DataLoader(valid_set, batch_size=128, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=128, shuffle=False)

    optimizer = Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    if task_type == "regression":
        loss_func = MSELoss()
    elif task_type == "classification":
        loss_func = BCELoss()
    
    best_model = None
    best_score = None
    wait = 0
    patient = 20

    for epoch in range(0, 100):
        step = 0
        tot_loss = 0
        for batch_data, labels in train_dataloader:
            batch_data = batch_data.to(device)
            labels = labels.to(device)
            model.train()
            with torch.enable_grad():
                preds = model(batch_data.to(device))
                loss = loss_func(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            step += 1
            tot_loss += loss
        # if epoch % 20 == 19:
        y_valid_pred = evaluate(model, valid_dataloader, task_type)

        if task_type == "regression":
            mse = mean_squared_error(y_valid, y_valid_pred)
            print(f"Epoch {epoch + 1}: tot_loss: {tot_loss / step}; mse: {mse}")
            score = -mse
        elif task_type == "classification":
            f1, acc = f1_score(y_valid, y_valid_pred), accuracy_score(y_valid, y_valid_pred)
            print(f"Epoch {epoch + 1}: tot_loss: {tot_loss / step}; f1: {f1}; acc: {acc}")
            score = f1
        if best_model is None or score > best_score:
            best_model = deepcopy(model)
            best_score = score
            wait = 0
        else:
            wait += 1
            if wait >= patient:
                break

    model = best_model
    # Make predictions
    y_train_pred = evaluate(model, train_dataloader, task_type)
    y_test_pred = evaluate(model, test_dataloader, task_type)
    
    # Calculate metrics
    if task_type == "regression":
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        result = {
            'model': model_name,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    else:
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        result = {
            'model': model_name,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'train_acc': train_acc,
            'test_acc': test_acc
        }
    
    print(result)
    exit(0)
    return result


def main():
    print("=" * 80)
    print("DCN on Boston Housing Dataset")
    print("=" * 80)
    
    # Load data
    if TASK == "regression":
        dataset = 'boston_housing'
    elif TASK == 'classification':
        dataset = 'bank-marketing'
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    
    print(f"\nLoading data from {data_path}")
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)
    print(f"Loaded {X_df.shape[0]} samples with {X_df.shape[1]} features")
    
    # Encode data (similar to RRL preprocessing)
    db_enc = DBEncoder(f_df, discrete=False, y_one_hot=False, drop=None)
    db_enc.fit(X_df, y_df, task_type=TASK)
    discrete_catenum = [len(x) for x in db_enc.feature_enc.categories_]
    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True, use_log=False)
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
        
        model = DCN(discrete_catenum, db_enc.continuous_flen, 512, 3, 3, TASK).to(device)
        result = train_model(model, X_train, y_train, X_test, y_test, "DCN", task_type=TASK)
            
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

    