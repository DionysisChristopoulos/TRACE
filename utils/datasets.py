import torch
from torch.utils.data import TensorDataset, random_split
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import json

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from ctgan import CTGAN

def labels2indices(df, label_names):
    if not isinstance(label_names, list):
        raise TypeError("label_names must be a list")
    return [df.columns.get_loc(label) for label in label_names]

def separate_features_by_index(tensor, numerical_indices, num_dtype=torch.float32, cat_dtype=torch.int32):
    if not isinstance(numerical_indices, list):
        raise TypeError("numerical_features must be a list")
    
    if len(numerical_indices) > 0:
        categorical_indices = [i for i in list(range(tensor.shape[1])) if i not in numerical_indices]

        tensor_num = tensor[:, numerical_indices].to(num_dtype)
        tensor_cat = tensor[:, categorical_indices].to(cat_dtype)
    else:
        tensor_num = None
        tensor_cat = tensor.to(cat_dtype)
    return tensor_cat, tensor_num

def separate_features_by_name(df, numerical_features, num_dtype=float, cat_dtype=int):
    if not isinstance(numerical_features, list):
        raise TypeError("numerical_features must be a list")
    
    if len(numerical_features) > 0:
        df_numerical = df[numerical_features].astype(num_dtype)
        df_categorical = df.drop(columns=numerical_features)
        df_categorical = df_categorical.astype(cat_dtype)
    else:
        df_numerical = None
        df_categorical = df.astype(cat_dtype)
    return df_categorical, df_numerical

def read_data(data_path, meta_path, target_label, encoded=False):
    df = pd.read_csv(data_path)
    data = df.drop(columns=[target_label], inplace=False) # drop target feature from final dataframe
    targets = pd.DataFrame({target_label: df[target_label]}) # target labels: 0: False 1: True
    
    with open(meta_path, 'r') as f:
        features = json.load(f)
    categorical_columns = list(features['categorical'].keys())

    if encoded:
        data = pd.get_dummies(data, columns=categorical_columns, prefix=categorical_columns, drop_first=False, dtype=int)
    
    for i, column in enumerate(data.columns):
        print(f"Column {i}: {column}")
    print(f"\nDataframe shape: {data.shape}")
    print(f"\nTarget: {targets.value_counts()}")
    return data, targets, features

def build_datasets(args, device='cpu'):
    x, y, feature_metadata = read_data(data_path=args.data.data_dir, 
                                       meta_path=args.data.metadata_dir, 
                                       target_label=args.data.target_label)
    
    num_indices = labels2indices(x, feature_metadata['continuous'])

    x, y = torch.tensor(x.values, device=device), torch.tensor(y.values, dtype=torch.float, device=device)
    X_train, X_test, Y_train, Y_test = train_test_split(x.cpu(), y.cpu(), test_size=(1-args.train.train_perc), stratify=y.cpu(), shuffle=True)

    if args.oversampling:
        
        if args.oversampling_mode == 'smote':
            oversample = SMOTE(sampling_strategy=args.smote.oversampling_strategy,
                                k_neighbors = args.smote.k_neighbors,
                                random_state=args.train.seed)
            X_train, Y_train = oversample.fit_resample(X_train, Y_train)

            if args.smote.undersampling:
                assert isinstance(args.smote.oversampling_strategy, float), "oversampling_strategy must be a float if undersampling is enabled"
                undersample = RandomUnderSampler(sampling_strategy=args.smote.undersampling_strategy, 
                                                    random_state=args.train.seed)
                X_train, Y_train = undersample.fit_resample(X_train, Y_train)

            X_train = torch.tensor(X_train, device=device)
            Y_train = torch.tensor(Y_train[..., np.newaxis], dtype=torch.float, device=device)
        
        elif args.oversampling_mode == 'ctgan':
            train_tensor = torch.cat((X_train, Y_train), dim=-1)
            
            target_label = args.data.target_label
            column_names = feature_metadata['continuous'] + list(feature_metadata['categorical'].keys())
            column_names.append(target_label)
            discrete_cols = column_names[len(num_indices):]
            
            df_train = pd.DataFrame(train_tensor.cpu().numpy(), columns=column_names)
            df_train_positive = df_train[df_train[target_label] == 1.]
            
            print(f"Number of positive samples: {len(df_train_positive)}")
            print(f"Number of discrete columns: {len(discrete_cols)}")

            ctgan = CTGAN(epochs=args.ctgan.epochs, verbose=True, cuda=True)
            ctgan.set_device(device)
            ctgan.fit(df_train_positive, discrete_cols)

            samples2gen = len(df_train[df_train[target_label] == 0.]) - len(df_train_positive)
            print(f"Number of samples (rows) to generate: {samples2gen}")
            synthetic_data = ctgan.sample(samples2gen, condition_column=target_label, condition_value=1.0)
            synthetic_data = torch.tensor(synthetic_data.values, dtype=torch.float, device=device)
            synthetic_data_x, synthetic_data_y = synthetic_data[:, :-1], synthetic_data[:, -1].unsqueeze(-1)

            X_train = torch.cat((X_train.to(device), synthetic_data_x), dim=0)
            Y_train = torch.cat((Y_train.to(device), synthetic_data_y), dim=0)
            shuffled_indices = torch.randperm(X_train.size(0))
            X_train = X_train[shuffled_indices]
            Y_train = Y_train[shuffled_indices]
    
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    return train_dataset, test_dataset, num_indices