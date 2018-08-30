import numpy as np


REG_PARAMS = {
    'MMS': {

        'STORE_SC_WEEK': {
            'LGB': {
                'objective': 'regression',
                'boosting': 'dart',
        #         'max_depth': 10,
                'num_leaves': 70,
                'min_data_in_leaf': 50,
                'max_cat_threshold': 200,
                'learning_rate': 0.05,
                'feature_fraction': 0.4,
                'bagging_fraction': 0.8,
                'bagging_freq': 10,
                'bagging_seed': 2**7,
                'drop_rate': 0.01,
                'uniform_drop': True,
                'max_drop': 10,
                'lambda_l2': 1.2,
                'metric': ['l2', 'l2_root', 'huber'],
                'save_binary': True,
                'num_threads': 16
            }
        },

        'REGION_SKC_WEEK': {
            'LGB': {
                'objective': 'regression',
                # 'max_depth': 10,
                'num_leaves': 70,
                'min_data_in_leaf': 150,
                'max_cat_threshold': 200,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 10,
                'bagging_seed': 2**7,
                'metric': ['l2', 'l2_root', 'huber'],
                'save_binary': True,
                'num_threads': 16
            }
        },

        'GBR_CV': {
            'n_estimators': np.arange(100, 500, 100),
            'loss': ["ls", "lad", "huber", "quantile"],
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'max_depth': range(3, 11),
            'min_samples_split': range(2, 6),
            'min_samples_leaf': range(1, 6),
            'subsample': np.arange(0.05, 1.01, 0.05),
            'max_features': np.arange(0.05, 1.01, 0.05),
            'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        },

        'GBR': {
            'n_estimators': 250,
            'loss': 'ls',
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_samples_splits': 2,
            'min_samples_leaf': 1,
            'subsample': 1.0,
            'max_features': None
        },

    },

    'MMS_STORE_DALIY_SALES_AMOUNT': {

    }
}


CLF_PARAMS = {
    'MMS': {
        'LGB': {
            'objective': 'multiclass',
    #         'max_depth': 10,
            'num_leaves': 70,
            'min_data_in_leaf': 150,
            'max_cat_threshold': 200,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'bagging_seed': 2**7,
            'metric': ['multi_logloss', 'multi_error', 'auc'],
            'save_binary': True,
            'num_threads': 16,
            'num_class': 4
        }
    }
}
