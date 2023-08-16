import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from src.models.pipeline import pipeline
import hydra
from hydra import utils
from omegaconf import DictConfig
import os

from src.visualization.visualize import conf_matrix, plot_roc_det

@hydra.main(version_base=None, config_path='../../config', config_name='config')
def train_model(config: DictConfig):

    # Get working directory
    current_path = utils.get_original_cwd() + '/'
    # Load Dataset
    dataset = pd.read_csv(current_path + config.data.raw,index_col='id')

    # Define target and features
    X = dataset.drop(config.variables.target,axis=1)
    y = dataset[config.variables.target]

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

    # Reset index
    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)

    # Save sampled datasets
    X_train.to_csv(current_path + config.data.interim.X_train, index=False)
    X_test.to_csv(current_path + config.data.interim.X_test, index=False)
    y_train.to_csv(current_path + config.data.interim.y_train, index=False)
    y_test.to_csv(current_path + config.data.interim.y_test, index=False)
    
    # Load the pipeline
    full_pipe = pipeline(config)
    
    print('Training in progress...')
    full_pipe.fit(X_train, y_train)
    
    preds = full_pipe.predict(X_test)
    conf_matrix(y_test,preds,path=config.reports.figures.matrix)
    print('Confusion matrix saved')

    classifiers = {f'{config.model.name}': full_pipe}
    plot_roc_det(classifiers,X_test,y_test,path=config.reports.figures.curves)
    print('Evaluation curves saved')

    print('             CLASIFICATION REPORT           ')
    print(classification_report(y_test,preds))
    joblib.dump(full_pipe, current_path + config.model.dir + config.model.name + '.joblib')


if __name__ == '__main__':
    train_model()
