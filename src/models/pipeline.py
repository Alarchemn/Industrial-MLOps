from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import src.models.preprocessors as pp
from omegaconf import OmegaConf

# Final pipeline for predictions and training 
# Check preprocessors.py for more info
def pipeline(config):

    # We need to convert from 'omegaconf.listconfig.ListConfig' to primite python list
    # Aparently Sklearn is not fully compatible with omegaconf.listconfig
    drop_ft = config.variables.drop_ft
    num_ft = OmegaConf.to_object(config.variables.num_ft)
    cat_ft = OmegaConf.to_object(config.variables.cat_ft)
    bin_ft = OmegaConf.to_object(config.variables.bin_ft)

    full_pipe = Pipeline(
        [
            ('clean_names', pp.CleanFeatureNames()),

            ('drop_features', pp.DropUnecessaryFeatures(drop_ft)),

            ('numerical', pp.NumericalPreprocessor(num_ft)),

            ('categorical', pp.CatBinPreprocessor(cat_ft=cat_ft, bin_ft=bin_ft)),

            ('new_features', pp.CreateNewFeatures()),

            ('clasifier', XGBClassifier(
                n_estimators = int(config.model.model_params['n_estimators']),
                max_depth = config.model.model_params['max_depth'],
                gamma = config.model.model_params['gamma'],
                colsample_bytree = config.model.model_params['colsample_bytree'],
                reg_lambda = config.model.model_params['reg_lambda'],
                min_child_weight = config.model.model_params['min_child_weight'],
                random_state = config.model.model_params['random_state']
                )
            )
        ]
    )

    return full_pipe