import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class Visualizer():

    @classmethod
    def distribution(df, col_name=None, kind='hist', layout=()):
        if col_name is None:
            raise Exception("Please provide col_name!")

        df[col_name].plot(kind=kind, figsize=(20, 12))


    @classmethod
    def corrmat(df, col_names=[]):
        if len(col_names) < 2:
            raise Exception("Please provide at least 2 col_names! e.g. ['col_1', 'col_2']")

        corr = df[col_name].corr()
        f, ax = plt.subplots(figsize=(10, 8))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(
            corr, cmap=cmap,
            center=0, linewidths=.25, vmax=.8, fmt='.2f',
            square=True, annot=True,
            annot_kws={'size': 7}, cbar_kws={"shrink": 0.6}
        )
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.show()


    # def pairplot(df, col_names=[], label_names=None):
    #     if len(col_names) < 2:
    #         raise Exception("Please provide at least 2 col_names! e.g. ['col_1', 'col_2']")
        
    #     sns.pairplot(
    #         df_s[col_names],
    #         diag_kind="kde",
    #         plot_kws=dict(s=50, edgecolor="b", linewidth=1),
    #         diag_kws=dict(shade=True),
    #         size=4
    #     )


    # def show_relationshipWithLabel(df, col_names=[], label_names=None, col_types='numeric'):
    #     if col_types == 'numeric':
    #         for col in col_names:
    #             fig, ax = plt.subplots()
    #             ax.scatter(df[col], df[label_names])
    #             plt.xlabel(col)
    #             plt.ylabel(label_names)
    #     elif col_types == 'categories':
    #         for col in col_names:
    #             data = pd.concat([df[label_names], df[col]], axis=1)
    #             f, ax = plt.subplots()
    #             fig = sns.boxplot(x=col, y=label_names, data=data)
    #     plt.show()
