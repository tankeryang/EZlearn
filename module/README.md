# mudule

> 自定义功能模块

## Usage

### TrainerBase

参考源码 Usage

### DataAnalyser

```python
import sys
import pandas as pd
sys.path.append("${path to ml-basic-platform}")
from module import DataAnalyser as da

df = pd.read_csv("some.csv")

# 查看数据详情
da.detials(df)

# 查看 target 数据分布
da.distribution(df, 'target')

# 获取数值型特征列
numeric_cols = []
numeric_cols = da.get_numeric_cols(df)

# 获取类别型特征列
categories_cols = []
categories_cols = da.get_categories_cols(df)
```

### FeatureGenerator

```python
import sys
import pandas as pd
sys.path.append("${path to ml-basic-platform}")
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

## FeatureSelector

参考: [Feature Selector Usage.](https://github.com/WillKoehrsen/feature-selector/blob/master/Feature%20Selector%20Usage.ipynb)

### Visualizer

参考源码

### RouteChain

```python
import sys
sys.path.append("${path to ml-basic-platform}")
from module import RouteChain

print(RouteChain().fuck.you.man)
# >>> /fuck/you/man
```

### DBUtiles

```python
import os
import sys
sys.append("${path to ml-basic-platform}")
from module import PrestoUtils

PRESTO_CONFIG_FILE = os.path.abspath("${path to presto config file}")

# 使用 config 文件初始化 PrestoUtils
pu = PrestoUtils(config_file=PRESTO_CONFIG_FILE)

# 使用参数初始化 PrestoUtils
pu2 = PrestoUtils(host='10.10.22.8', port='10300', user='prod', catalog='prod_hive')

# 查看 properties
print(pu.properties)
print(pu2.properties)

sql = "some sql script"
cur = pu.presto_conn.cursor()
cur.execute(sql)
```

### const

```python
from module import const

const.pi = 3.14
print(const.pi)

const.pi = 3.14159265358979323846264338327950288
# >>> 报错
# Traceback (most recent call last):
#   File "/Users/yang/workspace/PycharmProjects/FP-project/ml-basic-platform/test/module_test.py", line 9, in <module>
#     const.pi = 3.14159265358979323846264338327950288
#   File "/Users/yang/workspace/PycharmProjects/FP-project/ml-basic-platform/module/const.py", line 11, in __setattr__
#     raise self.ConstError("Can't rebind const(%s)" % name)
# module.const.ConstError: Can't rebind const(pi)
```