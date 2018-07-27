# ml-training-platform

## 1. project info

* __version__: `v1.0`
* __discription__: 机器学习通用开发框架

## 2. Usage

### 2.1. config

训练参数配置，presto连接配置，和其他所有的配置统一放于此

__训练参数请在`parameters.py`中进行配置__

### 2.2. training

训练脚本统一在`train/<项目名>/`下开发，关于如何编写训练脚本，参考`module/TrainerBase.py`的 Usage

### 2.3. module

`module`下有一些实现好的模块 (工具类)，具体使用直接参考源码即可

* __const.py__: python实现的常量类型(无用)
* __DataAnalyzer.py__: 数据分析工具类
* __DBUtils.py__: 数据库工具类
* __FeatureGenerator.py__: 特征生成工具类
* __FeatureSelector.py__: 特征分析&选择工具类
* __RouteChain.py__: 实现动态路由，用于 flask url 路由
* __TrainerBase.py__: 训练脚本基础类
* __Visualizer.py__: 可视化工具类

## Note

具体的目录结构参考`file_tree`