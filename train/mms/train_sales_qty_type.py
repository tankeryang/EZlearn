import os
import gc
import sys
import time
import logging
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sqlalchemy import create_engine
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, train_test_split
# customer module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from module import TrainerBase
from config import DTYPE_PARAMS, CLF_PARAMS, PROCESSOR
warnings.filterwarnings('ignore')


class Trainer(TrainerBase):

    def __init__(self):
        super(Trainer, self).__init__()
        self.__df_source_data = None
        self.__df_useful_data = None
        self.__df_train_test_data = {}
        self.__best_iteration = None
        self.model = None


    def get_source_data(self):
        if self.args.training_type == 'online':
            presto_engin = create_engine('presto://prod@10.10.22.8:10300/prod_hive/ads_mms')
            con = presto_engin.connect()

            if self.args.training_model == 'allocate':
                sql = """
                    select * from ads_mms.training_data_store_skc_week_merge
                    where store_level != 'D'
                    and store_retail_amount_mean_8weeks is not null
                    and store_sales_amount_mean_8weeks is not null
                    and interval_weeks_to_list = 0
                    and year_code >= '2017'
                """
            elif self.args.training_model == 'transform':
                sql = """
                    select * from ads_mms.training_data_store_skc_week_merge
                    -- where store_level != 'D'
                    where store_retail_amount_mean_8week is not null
                    and store_sales_amount_mean_8week is not null
                    and tempreture_day_highest is not null
                    and interval_weeks_to_list >= 1
                    and year_code >= '2017'
                """
            else:
                logging.error("Unknown argument {}, Please use [allocate | transform]".format(self.args.training_model))
                sys.exit(1)
            
            self.logger.info('='*60)
            self.logger.info("Loading source data from presto...")
            self.logger.info('='*60)

            # 抽数&数据类型压缩
            df_source_data = pd.read_sql_query(sql=sql, con=con)
            for col in DTYPE_PARAMS['MMS']:
                df_source_data[col] = df_source_data[col].astype(DTYPE_PARAMS['MMS'][col])

        elif self.args.training_type == 'offline':
            self.logger.info('='*60)
            self.logger.info("Loading source data from local csv file...")
            self.logger.info('='*60)

            df_source_data = pd.read_csv(
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'input', arg)),
                dtype=DTYPE_PARAMS['MMS'],
                low_memory=False
            )

        else:
            logging.error("Unknown argument {}, Please use [online | offline]".format(self.args.training_type))
            sys.exit(1)

        self.logger.info("Loading complete!")

        self.__df_source_data = df_source_data


    def get_useful_data(self):
        if self.__df_source_data is None:
            pass
        else:
            df_useful_data = self.__df_source_data.copy()

            if self.args.training_model == 'allocate':
                df_useful_data.drop(
                    columns=[
                        'year_code', 'main_cate',
                        'prev_year_week_code',
                        'interval_weeks_to_list',
                        'interval_weeks_to_prev',
                        'list_dates_year_code',
                        'list_dates_week_code',
                        'list_dates_year_week_code',
                        'discount_rate_mean_last_2week',
                        'discount_rate_mean_last_week',
                        'discount_rate_mean_change_rate',
                        'skc_con_sale_rate_last_week',
                        'skc_con_sale_rate',
                        'sales_qty_last_2week',
                        'sales_qty_last_week',
                        'sales_qty_last_week_and_last_2week_gap'
                    ],
                    inplace=True
                )

            elif self.args.training_model == 'transform':
                df_useful_data.drop(
                    columns=[
                        'year_code', 'main_cate',
                        'prev_year_week_code',
                        'list_dates_year_code',
                        'list_dates_week_code',
                        'list_dates_year_week_code'
                    ],
                    inplace=True
                )

            else:
                logging.error("Unknown argument {}, Please use [allocate | transform]".format(self.args.training_model))
                sys.exit(1)

        self.__df_useful_data = df_useful_data


    def feature_engineering(self):
        # print(df_useful_data.shape)
        # 数据范围压缩
        self.__df_useful_data['sales_qty'] = self.__df_useful_data['sales_qty'].clip(lower=0, upper=20)
        # df_useful_data['interval_weeks_to_list'] = df_useful_data['interval_weeks_to_list'].clip(lower=0)

        self.__df_useful_data.dropna(axis='index', subset=['tempreture_day_avg'], inplace=True)
        # df_useful_data.drop(
        #     index=df_useful_data.loc[df_useful_data['top_1_sub_cate_last_week'].isnull() == True, :].index,
        #     inplace=True
        # )

        # 小样本sms
        df_useful_data_1_2_sms, _ = train_test_split(
            self.__df_useful_data[self.__df_useful_data['sales_qty_type']=='1-2'],
            train_size=0.05, random_state=42
        )
        self.logger.info("type 1-2: {}".format(df_useful_data_1_2_sms.shape))
        del _
        gc.collect()

        df_useful_data_3_5_sms, _ = train_test_split(
            self.__df_useful_data[self.__df_useful_data['sales_qty_type']=='3-5'],
            train_size=0.2, random_state=42
        )
        self.logger.info("type 3-5: {}".format(df_useful_data_3_5_sms.shape))
        del _
        gc.collect()

        self.__df_useful_data.drop(
            index=self.__df_useful_data[
                (self.__df_useful_data['sales_qty_type']=='1-2') | (self.__df_useful_data['sales_qty_type']=='3-5')
            ].index,
            inplace=True
        )

        self.__df_useful_data = pd.concat([
            df_useful_data_1_2_sms, df_useful_data_3_5_sms, self.__df_useful_data
        ]).reset_index()
     
        print(self.__df_useful_data.shape)


    def preprocessing(self):
        self.logger.info('='*60)
        self.logger.info("Preprocessing df_useful_data...")
        self.logger.info('='*60)

        # 设置index
        df_data = self.__df_useful_data.copy()
        df_data.set_index(['store_code', 'skc_code'], inplace=True)

        # 小样本之后的步骤
        df_data.drop(columns=['index'], inplace=True)

        # 类型编码
        self.__le = PROCESSOR['LE']
        df_data['sales_qty_type'] = self.__le.fit_transform(df_data['sales_qty_type'])
        
        # 划分
        df_train = df_data[df_data['year_week_code'] < '201821']
        df_test = df_data[df_data['year_week_code'] >= '201821']

        df_train_y = df_train.pop('sales_qty_type')
        df_train.drop(columns=['year_week_code', 'sales_qty'], inplace=True)
        df_test_y = df_test.pop('sales_qty_type')
        df_test.drop(columns=['year_week_code', 'sales_qty'], inplace=True)
        df_data_y = df_data.pop('sales_qty_type')
        df_data.drop(columns=['year_week_code', 'sales_qty'], inplace=True)

        # 获得数值型和类别型数据
        # num_cols = get_numeric_cols(df_train)
        # cat_cols = get_categories_cols(df_train)
        
        # 标准化
        ## 保持DataFrame
        # df_train[num_cols] = PROCESSOR['RS'].fit_transform(df_train[num_cols])
        # df_test[num_cols] = PROCESSOR['RS'].fit_transform(df_test[num_cols])
        # df_data[num_cols] = PROCESSOR['RS'].fit_transform(df_data[num_cols])

        # 保存划分后的数据集
        self.__df_train_test_data['df_train_X'] = df_train
        self.__df_train_test_data['df_train_y'] = df_train_y
        self.__df_train_test_data['df_test_X'] = df_test
        self.__df_train_test_data['df_test_y'] = df_test_y
        self.__df_train_test_data['df_data_X'] = df_data
        self.__df_train_test_data['df_data_y'] = df_data_y

        # 打印日志
        self.logger.info('df_train_X shape: {}'.format(df_train.shape))
        self.logger.info('df_test_X shape: {}'.format(df_test.shape))
        self.logger.info('df_train_y shape: {}'.format(df_train_y.shape))
        self.logger.info('df_test_y shape: {}'.format(df_test_y.shape))

        self.logger.info("Preprocessing complete!")


    def test_model(self):
        self.logger.info('='*60)
        self.logger.info("Test model by df_train_X/y(train data), df_test_X/y(valid data)...")
        self.logger.info('='*60)

        # 加载数据至lgb.Dataset类型
        dtrain = lgb.Dataset(self.__df_train_test_data['df_train_X'], label=self.__df_train_test_data['df_train_y'])
        dtest = lgb.Dataset(self.__df_train_test_data['df_test_X'], label=self.__df_train_test_data['df_test_y'], reference=dtrain)

        # 训练
        bst = lgb.train(
            params=CLF_PARAMS['MMS']['LGB'],
            train_set=dtrain,
            valid_sets=[dtrain, dtest],
            num_boost_round=50,
            early_stopping_rounds=10,
            verbose_eval=5
        )

        # 打印指标
        y_pred_prop = bst.predict(self.__df_train_test_data['df_test_X'], num_iteration=bst.best_iteration or 1000)
        y_pred = []
        for prop_array in y_pred_prop:
            prop_list = prop_array.tolist()
            y_pred.append(prop_list.index(max(prop_list)))

        df_metrics = pd.DataFrame()
        df_metrics['y_true'] = self.__df_train_test_data['df_test_y']
        df_metrics['y_pred'] = y_pred

        self.logger.info("Evaluation by valid data:")

        for label in self.__df_train_test_data['df_test_y'].unique():
            df_metrics_label = df_metrics[df_metrics['y_true']==label]
            self.logger.info(
                "{} -- f1: {}".format(
                    list(self.__le.inverse_transform([label]))[0],
                    f1_score(df_metrics_label['y_true'], df_metrics_label['y_pred'], average='micro')
                )
            )

        self.logger.info(60*'-')

        fm_s = dict(zip(self.__df_train_test_data['df_train_X'].columns.tolist(), list(bst.feature_importance(importance_type='split'))))
        fm_g = dict(zip(self.__df_train_test_data['df_train_X'].columns.tolist(), list(bst.feature_importance(importance_type='gain'))))

        self.logger.info("feature importances: ")

        self.logger.info("split: ")
        for item in sorted(fm_s.items(), key=lambda d: d[1], reverse=True):
            self.logger.info("{}: {}".format(item[0], item[1]))
        self.logger.info(60*'-')

        self.logger.info("gain: ")
        for item in sorted(fm_g.items(), key=lambda d: d[1], reverse=True):
            self.logger.info("{}: {:.3f}".format(item[0], item[1]))
        self.logger.info(60*'-')

        self.__best_iteration = bst.best_iteration


    def get_model(self):
        self.logger.info('='*60)
        self.logger.info("Train by df_data_X/y(all training data) and save model...")
        self.logger.info('='*60)

        # 加载数据至lgb.Dataset类型
        dtrain = lgb.Dataset(self.__df_train_test_data['df_data_X'], label=self.__df_train_test_data['df_data_y'])

        bst = lgb.train(
            params=CLF_PARAMS['MMS']['LGB'],
            train_set=dtrain,
            num_boost_round=self.__best_iteration,
            valid_sets=[dtrain],
            verbose_eval=5
        )

        # 打印指标
        y_pred_prop = bst.predict(self.__df_train_test_data['df_test_X'], num_iteration=bst.best_iteration or 1000)
        y_pred = []
        for prop_array in y_pred_prop:
            prop_list = prop_array.tolist()
            y_pred.append(prop_list.index(max(prop_list)))

        df_metrics = pd.DataFrame()
        df_metrics['y_true'] = self.__df_train_test_data['df_test_y']
        df_metrics['y_pred'] = y_pred

        self.logger.info("Evaluation by valid data:")

        for label in self.__df_train_test_data['df_test_y'].unique():
            df_metrics_label = df_metrics[df_metrics['y_true']==label]
            self.logger.info(
                "{} -- f1: {}".format(
                    list(self.__le.inverse_transform([label]))[0],
                    f1_score(df_metrics_label['y_true'], df_metrics_label['y_pred'], average='micro')
                )
            )

        self.logger.info(60*'-')

        fm_s = dict(zip(self.__df_train_test_data['df_data_X'].columns.tolist(), list(bst.feature_importance(importance_type='split'))))
        fm_g = dict(zip(self.__df_train_test_data['df_data_X'].columns.tolist(), list(bst.feature_importance(importance_type='gain'))))

        self.logger.info("feature importances: ")

        self.logger.info("split: ")
        for item in sorted(fm_s.items(), key=lambda d: d[1], reverse=True):
            self.logger.info("{}: {}".format(item[0], item[1]))
        self.logger.info(60*'-')

        self.logger.info("gain: ")
        for item in sorted(fm_g.items(), key=lambda d: d[1], reverse=True):
            self.logger.info("{}: {:.3f}".format(item[0], item[1]))
        self.logger.info(60*'-')

        self.model = bst


    def save_model(self):
        if self.model is None:
            pass
        else:
            model_file_name = 'model_{}_{}_{}.txt'.format(
                self.args.solution_type, self.args.project, self.args.training_model
            )
            model_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output', model_file_name))

            self.model.save_model(model_file)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    # trainer.show_usage()
    # trainer.test()
