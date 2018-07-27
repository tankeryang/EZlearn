class DataAnalyzer():
    """
    数据分析工具类

    **Basic**

    这个类为实现数据分析的工具类，主要为在 jupyter 上做数据分析&探索时提供简洁快速的
    方法调用

    **Usage**

    直接看例子:
        ```
        import pandas as pd
        from module import DataAnalyzer as da

        df = pd.read_csv("some.csv")

        # 查看数据详情
        da.show_detials(df)

        # 查看 target 数据分布
        da.show_distribution(df, 'target')

        # 获取数值型特征列
        numeric_cols = []
        numeric_cols = da.get_numeric_cols(df)

        # 获取类别型特征列
        categories_cols = []
        categories_cols = da.get_categories_cols(df)
    """

    def __init__(self):
        pass


    @staticmethod
    def details(df):
        print('{:*^60}'.format('Data overview'))
        print(df.head())
        print('{:*^60}'.format('Data info'))
        print(df.info())
        print('{:*^60}'.format('Data describe'))
        print(df.describe().round(2).T)
        print('{:*^60}'.format('Data NAN info'))
        print(df.isnull().sum().sort_values(ascending=False))
        print('{:*^60}'.format('Data duplicated info'))
        print(df.duplicated().sum())
        print(60*'*')


    @staticmethod
    def distribution(df, label):
        print("skewness: %f" % df[label].skew())
        print("Kurtosis: %f" % df[label].kurt()) 


    @staticmethod
    def get_numeric_cols(df):
        numeric_cols = df.columns[(df.dtypes != 'object') & (df.dtypes != 'category')]
        return numeric_cols


    @staticmethod
    def get_categories_cols(df):
        categories_cols = df.columns[(df.dtypes == 'object') | (df.dtypes == 'category')]
        return categories_cols
