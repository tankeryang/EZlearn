import os
import sys
import time
import logging
import warnings
import argparse

warnings.filterwarnings('ignore')


class TrainerBase():
    """
    TrainerBase 训练脚本基类

    **Basic**

    这个类为实现不同问题类型 (classification, regression, clusting, ...)，不同训练类型 (online, offline)，
    不同训练模型 (model1, model2, ...)，不同项目 (mms, ...) 的训练脚本的基础框架类，他定义了基本的参数，和使用者
    需继承实现的方法

    **Usage**

    - 开发者须遵循以下规范:
        - 须在 `train/<项目名>/` 目录下编写训练脚本
        - 脚本名默认为 `train.py`，如有特殊的预测需求，则加后缀: `train_fuck.py`

    - 编写训练脚本:
        - 导入 TrainerBase 基类:
            ```
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
            from module import TrainerBase
            ```
        - 编写训练类 Trainer:
            ```
            class Trainer(TrainerBase):
                def __init__(self):
                    # 调用父类初始化方法
                    super(Trainer, self).__init__()

                    # 定义初始化的成员变量
                    ...
                
                # 实现基类定义的方法

            if __name__ == '__main__':
                trainer = Trainer()
                trainer.train()
            ```
        - 执行 (例子):
            ```
            python train.py \\
                --solution-type regression \\
                --training-type online \\
                --training-model transform \\
                --project mms
            ```
        
        具体需要实现的方法看下面源码

    .. version v1.0
    """

    NECESSARY_ARGS = {
        '--solution-type': 'solution_type',
        '--training-type': 'training_type',
        '--training-model': 'training_model',
        '--project': 'project'
    }


    USAGE = """
    python ${your_script_name}.py <option> [argument]

    for help
    --------
    python ${your_script_name}.py -h

    example
    -------
    python ${your_script_name}.py \\
        --solution-type regression \\
        --training-type online \\
        --training-model transform \\
        --project mms
    """


    def __init__(self):
        """
        初始化脚本参数，日志对象
        初始化时将参数通过 self.__set_args() 绑定到 self.__args 变量上，并执行参数检查
        """
        self.__args = self.__set_args()
        self.__check_args()
        self.__logger = self.__get_logger()


    def __set_args(self):
        """
        设置参数选项
        """
        paraser = argparse.ArgumentParser(prog="python36 train.py", description="")

        paraser.add_argument('--usage', action='store_true', dest='usage', default=False, help="show usage.")
        paraser.add_argument(
            '--solution-type',
            action='store', dest='solution_type', type=str, help="set solution type"
        )
        paraser.add_argument(
            '--training-type', action='store', dest='training_type', type=str, help="set training type"
        )
        paraser.add_argument(
            '--training-model', action='store', dest='training_model', type=str, help="set training model"
        )
        paraser.add_argument('--project', dest='project', type=str, help="set project belong to")

        args = paraser.parse_args()

        args_key = list(map(lambda kv: kv[0], args._get_kwargs()))
        args_value = list(map(lambda kv: kv[1], args._get_kwargs()))
        self.__args_dict = dict(zip(args_key, args_value))

        return args
        
    
    def __check_args(self):
        """
        参数检查
        """
        # check usage
        if self.__args.usage is True:
            self.show_usage()
            sys.exit(0)

        # check necesary arguments
        for necessary_arg in TrainerBase.NECESSARY_ARGS.values():
            if self.__args_dict[necessary_arg] is None:
                logging.error(
                    "Please provide all necessary option: {}".format([key for key in TrainerBase.NECESSARY_ARGS.keys()])
                )
                sys.exit(1)


    def __get_logger(self):
        """
        获取logger日志句柄
        """
        # 获取logfile
        logfile_name = '_'.join(['train', self.__args.solution_type, self.__args.training_model]) + '.log'
        logfile = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'log', self.__args.project, logfile_name)
        )
        if not os.path.exists(logfile):
            f = open(logfile, 'w+')
            f.close()

        ## 定义Logger对象
        logger = logging.getLogger()
        logger.setLevel(level=logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        ## 获取文件logger句柄
        fh = logging.FileHandler(logfile, mode='a')
        fh.setLevel(level=logging.INFO)
        fh.setFormatter(formatter)  
        ## 获取终端logger句柄
        ch = logging.StreamHandler()  
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        ## 添加至logger 
        logger.addHandler(fh)  
        logger.addHandler(ch)

        return logger


    # 定义getter，使得子类可以访问定义的私有变量，但是子类要显式调用TrainerBase的__init__()方法

    @property
    def args(self):
        return self.__args


    @property
    def args_dict(self):
        return self.__args_dict


    @property
    def logger(self):
        return self.__logger


    # 定义子类需要实现的方法

    def get_source_data(self):
        """
        导入源数据至内存
        :params :
        :return : 源数据的 DataFrame [type:pd.DataFrame | name:df_source_data]
        """
        raise NotImplementedError("Not implemented!")


    def get_useful_data(self):
        """
        截取特征列
        :params df_source_data: 源数据 DataFrame [type:pd.DataFrame | from:get_source_data()]
        :return df_useful_data: 截取特征列后的 DataFrame [type:pd.DataFrame | name:df_useful_data]
        """
        raise NotImplementedError("Not implemented!")


    def feature_engineering(self):
        """
        特征工程
        :params df_useful_data: 截取特征列后的样本数据 [type:pd.DataFrame | from:get_useful_data()]
        :return df_useful_data: 特征工程后的样本数据 [type:pd.DataFrame | name:df_useful_data]
        """
        raise NotImplementedError("Not implemented!")


    def preprocessing(self):
        """
        数据预处理和划分训练测试集
        :params df_useful_data: 特征工程后的样本数据 [type:pd.DataFrame | from:get_useful_data()]
        :return df_train_X: 训练集特征DataFrame [type:pd.DataFrame]
        :return df_train_y: 训练集标签DataFrame [type:pd.DataFrame]
        :return df_test_X: 测试集特征DataFrame [type:pd.DataFrame]
        :return df_test_y: 测试集标签DataFrame [type:pd.DataFrame]
        :return df_data_X: 全量样本特征DataFrame [type:pd.DataFrame]
        :return df_data_y: 全量样本标签DataFrame [type:pd.DataFrame]

        .. note:
            在实现此方法时，你需要将上述的 return 值存在子类的一个成员字典中:
            ```
            self.__df_train_test_data['df_train_X'] = df_train
            self.__df_train_test_data['df_train_y'] = df_train_y
            self.__df_train_test_data['df_test_X'] = df_test
            self.__df_train_test_data['df_test_y'] = df_test_y
            self.__df_train_test_data['df_data_X'] = df_data
            self.__df_train_test_data['df_data_y'] = df_data_y
            ```    
        """

        # TODO:
        # 设置 index
        # 划分
        # 获得数值型和类别型的数据
        # 标准化

        raise NotImplementedError("Not implemented!")


    def test_model(self):
        """
        验证模型
        :params df_train_X: 训练集特征DataFrame [type:pd.DataFrame | from:preprocessing()]
        :params df_train_y: 训练集标签DataFrame [type:pd.DataFrame | from:preprocessing()]
        :params df_test_X: 测试集特征DataFrame [type:pd.DataFrame | from:preprocessing()]
        :params df_test_y: 测试集标签DataFrame [type:pd.DataFrame | from:preprocessing()]
        """

        # TODO:
        # 加载数据至模型数据对象，如 lgb.Dataset
        # 训练，返回 bst 模型
        # 打印指标，特征重要性

        raise NotImplementedError("Not implemented!")


    def get_model(self):
        """
        全量训练模型
        :params df_data_X: 全量样本特征[type:pd.DataFrame | from:preprocessing()]
        :params df_data_y: 全量样本标签[type:pd.DataFrame | from:preprocessing()]
        :return bst: 最终模型
        """

        # TODO:
        # 加载数据至模型数据对象，如 lgb.Dataset
        # 训练，返回 bst 模型
        # 打印指标，特征重要性

        raise NotImplementedError("Not implemented!")


    def save_model(self):
        """
        保存模型
        """
        raise NotImplementedError("Not implemented!")


    def train(self):
        """
        训练主函数
        """
        self.get_source_data()
        self.get_useful_data()
        self.feature_engineering()
        self.preprocessing()
        self.test_model()
        self.get_model()
        self.save_model()
        
        self.__logger.info("Finish.")
        self.__logger.info(60*"=" + "\n")


    def show_usage(self):
        if self.__args.usage is True:
            print(TrainerBase.USAGE)


    def test(self):
        print(self.args_dict)
        print(os.path.dirname(__file__))
