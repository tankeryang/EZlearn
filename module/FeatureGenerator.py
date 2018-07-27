import os
import sys
import gc
import pandas as pd
import numpy as np
from DataAnalyzer import DataAnalyzer as da


class FeatureGenerator():
    """
    特征生成工具类

    **Basic**

    这个类为对数据生成统计量特征的工具类，主要为在 jupyter 上做模型预训练时提供简洁快速的
    方法调用

    **Usage**

    比如现在有一份数据 `some.csv` 如下:
        -----------------------
         class | name  | score
        -------+-------+-------
         A     | sb    |     0
         A     | lowb  |    59
         B     | dalao |    99
         C     | shen  |   100
         ----------------------
    
    你可以做如下操作:
        ```
        import pandas as pd
        from module import FeatureGenerator as fg

        df_source_data = pd.read_csv("some.csv")

        df_useful_data = fg.gen_count(df_source_data, ['class'])
        # <=> select count(*) from some.csv group by class;

        df_useful_data = fg.gen_count_unique(df_source_data, ['class'], 'name')
        # <=> select count(name) from some.csv group by class;

        df_useful_data = fg.gen_mean(df_source_data, ['class'], 'score')
        # <=> select avg(score) from some.csv group by class

        gen_feature = fg.get_gen_feature()
        ```

    .. note:
        仅作数据探索&数据分析&模型预训练用，实际环境特征构建主要用 SQL 实现
    """

    __gen_feature = []


    @classmethod
    def gen_count(
        cls, df, groupby_cols, agg_type=np.int16,
        show_result=False, show_detials=False, show_sql=True
    ):
        agg_name='{}-count'.format('_'.join(groupby_cols))

        if show_sql:
            print(
                "same as sql: select count(*) as {agg_name} from data group by {groupby_cols}".format(
                    agg_name=agg_name, groupby_cols=str(groupby_cols).strip('[').strip(']')
                )
            )

        gp = df[groupby_cols].groupby(groupby_cols).size().rename(agg_name).to_frame().reset_index()
        df = df.merge(gp, on=groupby_cols, how='left')

        if show_result:
            print(gp.sort_values(by=[agg_name], ascending=False))

        if show_detials:
            da.show_details(gp)

        df[agg_name] = df[agg_name].astype(agg_type)
        cls.__gen_feature.append(agg_name)

        return df


    @classmethod
    def gen_count_unique(
        cls, df, groupby_cols, counted_col, agg_type=np.int16,
        show_result=False, show_detials=False, show_sql=True
    ):
        agg_name = '{}-by-{}_count_unique'.format(('_'.join(groupby_cols)),(counted_col))

        if show_sql:
            print(
                "same as sql: select count({counted_col}) from data group by {groupby_cols}".format(
                    counted_col=counted_col, groupby_cols=str(groupby_cols).strip('[').strip(']')
                )
            )

        gp = df[groupby_cols+[counted_col]].groupby(groupby_cols)[counted_col].nunique().reset_index().rename(columns={counted_col:agg_name})
        df = df.merge(gp, on=groupby_cols, how='left')

        if show_result:
            print(gp.sort_values(by=[agg_name], ascending=False))

        if show_detials:
            da.show_details(gp)

        df[agg_name] = df[agg_name].astype(agg_type)
        cls.__gen_feature.append(agg_name)

        return df


    @classmethod
    def gen_cumcount(
        cls, df, groupby_cols, counted_col, agg_type=np.int16,
        show_result=False, show_detials=False, show_sql=True
    ):
        agg_name= '{}-by-{}_cumcount'.format(('_'.join(groupby_cols)),(counted_col))

        if show_sql:
            print("same as sql: select {groupby_cols}, '{counted_col}' row_number() over(partition by {groupby_cols}) from data".format(groupby_cols=str(groupby_cols).strip('[').strip(']'), counted_col=counted_col))

        gp = df[groupby_cols+[counted_col]].groupby(groupby_cols)[counted_col].cumcount()
        df[agg_name]=gp.values

        if show_result:
            print(gp.sort_values(by=[agg_name], ascending=False))

        if show_detials:
            da.show_details(gp)

        df[agg_name] = df[agg_name].astype(agg_type)
        cls.__gen_feature.append(agg_name)

        return df


    @classmethod
    def gen_mean(
        cls, df, groupby_cols, counted_col, agg_type=np.float32,
        show_result=False, show_detials=False, show_sql=True
    ):
        agg_name= '{}-by-{}_mean'.format(('_'.join(groupby_cols)),(counted_col))

        if show_sql:
            print(
                "same as sql: select avg({counted_col}) from data group by {groupby_cols}".format(
                    counted_col=counted_col, groupby_cols=str(groupby_cols).strip('[').strip(']')
                )
            )

        gp = df[groupby_cols+[counted_col]].groupby(groupby_cols)[counted_col].mean().reset_index().rename(columns={counted_col: agg_name})
        df = df.merge(gp, on=groupby_cols, how='left')

        if show_result:
            print(gp.sort_values(by=[agg_name], ascending=False))

        if show_detials:
            da.show_details(gp)

        df[agg_name] = df[agg_name].astype(agg_type)
        cls.__gen_feature.append(agg_name)

        return df


    @classmethod
    def gen_variance(
        cls, df, groupby_cols, counted_col, agg_type=np.float32,
        show_result=False, show_detials=False, show_sql=True
    ):
        agg_name= '{}-by-{}_variance'.format(('_'.join(groupby_cols)),(counted_col))

        if show_sql:
            print(
                "same as sql: select variance({counted_col}) from data group by {groupby_cols}".format(
                    counted_col=counted_col, groupby_cols=str(groupby_cols).strip('[').strip(']')
                )
            )

        gp = df[groupby_cols+[counted_col]].groupby(groupby_cols)[counted_col].var().reset_index().rename(columns={counted_col:agg_name})
        df = df.merge(gp, on=groupby_cols, how='left')

        if show_result:
            print(gp.sort_values(by=[agg_name], ascending=False))

        if show_detials:
            da.show_details(gp)

        self.__gen_feature.append(agg_name)

        return df


    @classmethod
    def get_gen_feature(cls):
        return cls.__gen_feature
