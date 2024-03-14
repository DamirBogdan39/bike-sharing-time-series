# Demand forecasting using bike sharing dataset

Project Organization
------------
    ├── configs
    │   └── feature_params     <- Configs for features
    │   └── path_config        <- Configs for all needed paths
    │   └── model_hparams      <- Hyperparameter confings after optimization
    │   └── param_grid         <- Hyperparameter values for grid search
    │   └── predict_config.yaml   <- Config for prediction pipline
    │   └── train_config.yaml   <- Config for train pipline
    │   └── optimizer_conf.yaml   <- Config for hyperparameter tuning 

    ├── data
    │   └── raw            <- The original, immutable data dump.
    │   └── interim        <- Train and test data.
    │
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts for splitting dataset to train and test
    │   │   └── make_dataset.py
    │   │
    │   ├── entities       <- Scripts for creating dataclasses
    │   │    
    │   │
    │   ├── features              <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │   └── optimize_model.py
    |   |
    │   ├── outputs       <- Hydra logs
    │   │   
    │   ├──  utils        <- Scripts for serialized models, reading data
    │   |    └── utils.py
    |   |
    |   ├── predict_pipeline.py   <- pipeline for making predictions
    |   |
    |   └── train_pipeline.py     <- pipeline for model training
    |   └── model_params_optimizer.py     <- pipeline for hyperparameter tuning  
    |   └── create_datasets.py     <- splitting the raw dataset into train and test
    |
    ├── tests              <- tests for the project
    ├── LICENSE
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── README.md          <- The top-level README for developers using this project.
