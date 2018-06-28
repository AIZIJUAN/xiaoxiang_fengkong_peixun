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
"""
对数值型特征进行特征处理
"""
class NumericFeatureEncoding:
    def __init__(self, features, data):
        self.numericFeature = features
        self.sampleData = data

    # 对数字变量进行缺失值填充
    #入参：x-原始值，replacement-替代值
    #返回值：该位置缺失值填充后的值
    @staticmethod
    def MakeupMissingNumerical(x, replacement):
        if np.isnan(x):
            return replacement
        else:
            return x

    @staticmethod
    def feature_encoding_process(numericFeature, sampleData):
        # 对数值型数据的缺失进行补缺
        #ProsperRating (numeric)特征的缺失值用0填充
        sampleData['ProsperRating (numeric)'] = sampleData['ProsperRating (numeric)'].map(
            lambda x: NumericFeatureEncoding.MakeupMissingNumerical(x, 0))
        #ProsperScore特征的缺失值用0填充
        sampleData['ProsperScore'] = sampleData['ProsperScore'].map(lambda x: NumericFeatureEncoding.MakeupMissingNumerical(x, 0))

        #计算全部训练样本DebtToIncomeRatio特征的均值
        avgDebtToIncomeRatio = np.mean(sampleData['DebtToIncomeRatio'])
        #DebtToIncomeRatio特征的缺失值用均值进行填充
        sampleData['DebtToIncomeRatio'] = sampleData['DebtToIncomeRatio'].map(
            lambda x: NumericFeatureEncoding.MakeupMissingNumerical(x, avgDebtToIncomeRatio))
        return sampleData

if __name__ == '__main__':
    print('test')
    #NumericFeatureEncoding.feature_encoding_process()
