import gradio as gr
import pandas as pd
from pathlib import Path
import random
import joblib

BASE_DIR = Path(__file__).resolve(strict=True).parent

df = pd.read_csv(f'{BASE_DIR}/train.csv')

unique_product_id = sorted(df['Product ID'].unique())
unique_type = ['H', 'L', 'M']
columns_names = ['Product ID', 'Type', 'Air temperature [K]',
       'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
       'Tool wear [min]','TWF', 'HDF', 'PWF', 'OSF',
       'RNF']

def predict(*args):
    
    args = list(args)
    
    for i in range(7,12):
        if args[i] == True:
            args[i] = 1
        else:
            args[i] = 0
            
    df = pd.DataFrame(data=[args],columns=columns_names)
    full_pipe = joblib.load(filename=f'{BASE_DIR}/XGBoost.joblib')
    preds = full_pipe.predict_proba(df)
    
    result = {
        'Correct': float(preds[0][0]),
        'Failure': float(preds[0][1])
    }
    return result