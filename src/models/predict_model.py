import hydra
from hydra import utils
from omegaconf import DictConfig
import joblib

# For test
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.visualization.visualize import conf_matrix, plot_roc_det

#@hydra.main(version_base=None, config_path='.', config_name='config')
def predict_model(config: DictConfig, input_data):

    # Get working directory
    current_path = utils.get_original_cwd() + '/'
    # Load Model
    full_pipe = joblib.load(filename=current_path + config.model.dir + config.model.name + '.joblib')

    # Make predictions
    preds = full_pipe.predict(input_data)

    return preds


@hydra.main(version_base=None, config_path='../../config', config_name='config')
def test_predict(config: DictConfig):

    # Get working directory
    current_path = utils.get_original_cwd() + '/'
    
    # Load Dataset
    X_test = pd.read_csv(current_path + config.data.interim.X_test)
    y_test = pd.read_csv(current_path + config.data.interim.y_test)

    # Make predictions with the X_test
    preds = predict_model(config, X_test)

    print('                     CLASIFICATION REPORT                     ')
    print('-' * 55)
    print(classification_report(y_test,preds))
    print('-' * 55)

    return preds

    

if __name__ == '__main__':
    test_predict()