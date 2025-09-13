# dataset loading and preprocessing

import torch
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_wine_data():
    wine_data = load_wine(as_frame=True)
    wine_df = wine_data.frame
    X = wine_df.drop('target', axis=1).values
    y = wine_df['target'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test
