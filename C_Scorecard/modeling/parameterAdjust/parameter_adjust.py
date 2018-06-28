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

class ParameterAdjust:
    def __init__(self, X, y):
        self.featureX = X
        self.labelY = y

    '''
        对基于CART的随机森林的调参，主要有：
        1，树的个数
        2，树的最大深度
        3，内部节点最少样本数与叶节点最少样本数
        4，特征个数
        此外，调参过程中选择的误差函数是均值误差，5倍折叠
    '''
    @staticmethod
    def gridSearchCVParameterAdjust(X, y, numFeatures2):
        # 从10开始，到80，步长为5
        #'n_estimators'：整型，默认值为10，表示森林中树的数量
        param_test1 = {'n_estimators': range(10, 80, 5)}
        #estimator：所使用的分类器
        #param_grid：字典或列表，表示需要最优化的参数的取值
        #scoring：准确度评价标准
        #cv：交叉验证参数
        gsearch1 = GridSearchCV(
            #min_samples_split：整型或浮点型，默认值为2，若为整型，表示分割内部节点需要的最小样本数，若为浮点型，表示百分比
            #min_samples_leaf：整型或浮点型，默认值为1，若为整型，表示在叶节点上所需的最小样本数，若为浮点型，表示百分比
            #max_depth：整型或None，默认值为None，若为整型，表示树的最大深度，若为None，表示节点会被扩展，直到所有叶子节点是纯的或者所有叶子结点包含的样本数少于min_samples_split samples
            #max_features：整型或浮点型或字符串型或None，默认值为“auto”，若为整型，表示特征数量，若为浮点型，表示百分比，若为“auto”，表示max_features=n_features，若为“sqrt”，表示max_features=sqrt(n_features)，若为“log2”，表示max_features=log2(n_features)，若为None，表示max_features=n_features
            #random_state：整型或None，默认值为None
            estimator=RandomForestRegressor(min_samples_split=50, min_samples_leaf=10, max_depth=8, max_features='sqrt',
                                            random_state=10),
            param_grid=param_test1, scoring='neg_mean_squared_error', cv=5)
        #运行网格搜索
        gsearch1.fit(X, y)
        #best_params_：描述了已取得最佳结果的参数的组合
        #best_score_：成员提供优化过程期间观察到的最好的评分
        gsearch1.best_params_, gsearch1.best_score_
        #从best_params_中取出‘n_estimators’保存在best_n_estimators中
        best_n_estimators = gsearch1.best_params_['n_estimators']

        #同理，使用网格搜索得出‘max_depth’和‘min_samples_split’的最佳取值
        param_test2 = {'max_depth': range(3, 21), 'min_samples_split': range(10, 100, 10)}
        gsearch2 = GridSearchCV(
            estimator=RandomForestRegressor(n_estimators=best_n_estimators, min_samples_leaf=10, max_features='sqrt',
                                            random_state=10, oob_score=True),
            param_grid=param_test2, scoring='neg_mean_squared_error', cv=5)
        gsearch2.fit(X, y)
        gsearch2.best_params_, gsearch2.best_score_
        best_max_depth = gsearch2.best_params_['max_depth']
        best_min_sample_split = gsearch2.best_params_['min_samples_split']

        #同理，使用网格搜索得出‘min_samples_split’和‘min_samples_leaf’的最佳取值
        param_test3 = {'min_samples_split': range(50, 201, 10), 'min_samples_leaf': range(1, 20, 2)}
        gsearch3 = GridSearchCV(
            estimator=RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth,
                                            max_features='sqrt',
                                            random_state=10, oob_score=True),
            param_grid=param_test3, scoring='neg_mean_squared_error', cv=5)
        gsearch3.fit(X, y)
        gsearch3.best_params_, gsearch3.best_score_
        best_min_samples_leaf = gsearch3.best_params_['min_samples_leaf']
        best_min_samples_split = gsearch3.best_params_['min_samples_split']

        #同理，使用网格搜索得出‘max_features’的最佳取值
        numOfFeatures = len(numFeatures2)
        mostSelectedFeatures = numOfFeatures / 2
        param_test4 = {'max_features': range(3, numOfFeatures + 1)}
        gsearch4 = GridSearchCV(
            estimator=RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth,
                                            min_samples_leaf=best_min_samples_leaf,
                                            min_samples_split=best_min_samples_split, random_state=10,
                                            oob_score=True),
            param_grid=param_test4, scoring='neg_mean_squared_error', cv=5)
        gsearch4.fit(X, y)
        gsearch4.best_params_, gsearch4.best_score_
        best_max_features = gsearch4.best_params_['max_features']

        print('best params: ')
        print(best_n_estimators, best_max_depth, best_min_samples_leaf, best_min_samples_split, best_max_features)
        return best_n_estimators, best_max_depth, best_min_samples_leaf, best_min_samples_split, best_max_features

if __name__ == '__main__':
    print('test')
    #ParameterAdjust.gridSearchCVParameterAdjust()