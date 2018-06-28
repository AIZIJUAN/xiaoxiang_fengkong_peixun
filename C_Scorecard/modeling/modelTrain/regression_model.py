# -*- coding: utf-8 -*

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from time import strptime,mktime
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation, metrics

class RegressionModel:
    def __init__(self):
        print('test')

    @staticmethod
    def RandomForestRegressorTrain(best_n_estimators,
                                   best_max_depth,
                                   best_min_samples_leaf,
                                   best_min_samples_split,
                                   best_max_features,
                                   X,
                                   y):
        cls = RandomForestRegressor(n_estimators=best_n_estimators,  # 森林中树的数量
                                    max_depth=best_max_depth,  #树的最大深度
                                    min_samples_leaf=best_min_samples_leaf,  #叶节点上所需的最小样本数
                                    min_samples_split=best_min_samples_split,  #分割内部节点需要的最小样本数
                                    max_features=best_max_features,  #特征数量
                                    random_state=10,  #随机器对象
                                    oob_score=True)  #是否计算袋外得分
        # 根据训练数据X、y构建随机森林
        cls.fit(X, y)
        return cls

    @staticmethod
    def RandomForestRegressorTrainTest(cls, sampleData, numFeatures2):
        # 预测trainData[numFeatures2]的回归值
        sampleData['pred'] = cls.predict(sampleData[numFeatures2])
        sampleData['less_rr'] = sampleData.apply(lambda x: int(x.pred > x.rec_rate), axis=1)
        print('mean less_rr:')
        print(np.mean(sampleData['less_rr']))
        err = sampleData.apply(lambda x: np.abs(x.pred - x.rec_rate), axis=1)
        print('mean err:')
        print(np.mean(err))
