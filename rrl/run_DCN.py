import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score, recall_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import warnings
from dcn.dcn import DCN
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import MSELoss, BCELoss
from torch.optim import Adam
import torch
from copy import deepcopy
from time import sleep
import random
warnings.filterwarnings('ignore')

from rrl.utils import read_csv, DBEncoder

TASK = "regression"
if TASK == "regression":
    DATA_DIR = '../data/boston_housing/'
elif TASK == 'classification':
    DATA_DIR = '../data/bank_marketing/'

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED) 
torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model, dataloader):
    y_pred = []
    for batch_data, labels in dataloader:
        model.eval()
        batch_data = batch_data.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            preds = model(batch_data)
        if TASK == "classification":
            preds = (preds >= 0.5).long()
        y_pred.append(preds)
    return torch.cat(y_pred, axis=0).cpu()

def call_result(y_train, y_train_pred, y_test, y_test_pred):
    if TASK == "regression":
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        result = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    elif TASK == "classification":
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_macro_f1 = f1_score(y_test, y_test_pred, average='macro')
        test_weighted_f1 = f1_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred)

        result = {
            'train_f1': train_f1,
            'test_f1': test_f1,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_macro_f1': test_macro_f1,
            'test_weighted_f1': test_weighted_f1,
            'test_recall': test_recall
        }
    return result

def train_model(model, X_train, y_train, X_test, y_test):
    """Train and evaluate a regression model"""
    # Train the model
    
    valid_ratio = 0.95
    sp = int(X_train.shape[0] * valid_ratio)
    X_train, X_valid = X_train[:sp], X_train[sp:]
    y_train, y_valid = y_train[:sp], y_train[sp:]
    train_set = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.float32)).view(-1, 1))
    valid_set = TensorDataset(torch.tensor(X_valid.astype(np.float32)), torch.tensor(y_valid.astype(np.float32)).view(-1, 1))
    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test.astype(np.float32)).view(-1, 1))
    train_dataloader = DataLoader(train_set, batch_size=512, shuffle=False)
    valid_dataloader = DataLoader(valid_set, batch_size=512, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=512, shuffle=False)

    optimizer = Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    if TASK == "regression":
        loss_func = MSELoss()
    elif TASK == "classification":
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
        y_valid_pred = evaluate(model, valid_dataloader)

        if TASK == "regression":
            mse = mean_squared_error(y_valid, y_valid_pred)
            print(f"Epoch {epoch + 1}: tot_loss: {tot_loss / step}; mse: {mse}")
            score = -mse
        elif TASK == "classification":
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
    y_train_pred = evaluate(model, train_dataloader)
    y_test_pred = evaluate(model, test_dataloader)
    
    result = call_result(y_train, y_train_pred, y_test, y_test_pred)
    print(result)
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
        
        model = DCN(discrete_catenum, db_enc.continuous_flen, 128, 3, 3, TASK).to(device)
        result = train_model(model, X_train, y_train, X_test, y_test)
        
        # model = GradientBoostingRegressor(n_estimators=200, random_state=SEED)
        # model = GradientBoostingClassifier(n_estimators=200, random_state=SEED)
        # model.fit(X_train, y_train)
        # y_train_pred = model.predict(X_train)
        # y_test_pred = model.predict(X_test)
        # result = call_result(y_train, y_train_pred, y_test, y_test_pred)

        results.append(result)
    
    mean_result = {}
    for key in results[0].keys():
        mean_result[key] = sum([x[key] for x in results]) / len(results)
    print(mean_result)


if __name__ == '__main__':
    main()

    