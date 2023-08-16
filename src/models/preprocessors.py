import pandas as pd
import xgboost
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Drop useless features
class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        self.variables = variables
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.drop(self.variables, axis=1)
        
        return X

# Replace NaN with mean and apply a standar scaler transformation
class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        self.variables = variables
        
    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

        return self
    
    def transform(self, X):
        X = X.copy()

        X[self.variables] = self.imputer.fit_transform(X[self.variables])
        X[self.variables] = self.scaler.fit_transform(X[self.variables])
        
        return X    

# Replace NaN with the mode and encode categorical features
class CatBinPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, cat_ft=None, bin_ft=None):
        self.cat_ft = cat_ft
        self.bin_ft = bin_ft
        
    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy='most_frequent')
        self.encoder = OrdinalEncoder()
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X[self.cat_ft + self.bin_ft] = self.imputer.fit_transform(X[self.cat_ft + self.bin_ft])
        X[self.cat_ft] = self.encoder.fit_transform(X[self.cat_ft])
        
        for feat in self.bin_ft:
            X[feat] = X[feat].astype('int')
        
        return X
    
# Create usefull features
class CreateNewFeatures(BaseEstimator, TransformerMixin):
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X['Temperature ratio'] = X['Process temperature K'] * X['Air temperature K']
        X['Mechanical ratio'] = X['Torque Nm'] * X['Rotational speed rpm']
        
        return X
    
# XGboost needs this
class CleanFeatureNames(BaseEstimator, TransformerMixin):
        
    def fit(self, X, y=None):
        self.columns = X.columns
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        new_columns = []
        for name in self.columns:
            new_columns.append(
                name.replace('[','').replace(']',''))
            
        X.columns = new_columns
        return X