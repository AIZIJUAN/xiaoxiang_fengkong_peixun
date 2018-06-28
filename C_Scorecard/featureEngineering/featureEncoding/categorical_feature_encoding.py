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
对类别型特征进行特征处理
"""
class CategoricalFeatureEncoding:
    def __init__(self, features, data):
        self.categoricalFeatures = features
        self.sampleData = data

    # 对类型字段进行缺失值填充
    #入参：x-原始值
    #返回值：该位置缺失值填充后的值
    @staticmethod
    def MakeupMissingCategorical(x):
        if str(x) == 'nan':
            return 'Unknown'
        else:
            return x

    @staticmethod
    def feature_encoding_process(categoricalFeatures, sampleData):
        '''
        类别型变量需要用目标变量的均值进行编码
        '''
        # 保存生成的编码特征
        encodedFeatures = []
        #保存编码特征取值的字典
        encodedDict = {}
        for var in categoricalFeatures:
            #对类别型特征进行缺失值填充
            sampleData[var] = sampleData[var].map(CategoricalFeatureEncoding.MakeupMissingCategorical)
            #关于var特征，对rec_rate进行group by操作，计算当前特征每个取值对应的rec_rate的均值
            avgTarget = sampleData.groupby([var])['rec_rate'].mean()
            #将avgTarget转换成字典形式，key为特征取值，value为当前取值对应的rec_rate的均值
            avgTarget = avgTarget.to_dict()
            #根据当前特征生成新特征
            newVar = var + '_encoded'
            #给新特征赋值
            sampleData[newVar] = sampleData[var].map(avgTarget)
            #将newVar加入到encodedFeatures中
            encodedFeatures.append(newVar)
            #encodedDict中加入var以及对应的字典类型的avgTarget
            encodedDict[var] = avgTarget
        return sampleData, encodedFeatures

if __name__ == '__main__':
    print('test')
    #CategoricalFeatureEncoding.feature_encoding_process()
