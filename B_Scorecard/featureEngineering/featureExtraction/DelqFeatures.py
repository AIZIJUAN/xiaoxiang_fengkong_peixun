import  pandas as pd
import  numpy as np
import  pickle
import  random
import  datetime
import  matplotlib.pyplot as pt
import  seaborn as sns
from sklearn.model_selection import train_test_split
from  statsmodels.stats.outliers_influence import variance_inflation_factor
import  statsmodels.api as sm

from settings import  *

train_data = pd.read_csv(ROOT_DIR + 'trainData.csv')

'''
逾期类型的特征在行为评分卡（预测违约行为）中，一般是非常显著的变量。
通过设定时间窗口，可以衍生以下类型的逾期变量：
'''
allFeatures = []
class DelqFeatureExtractor:
    def __init__(self):

        self.result_file = 'delqFeatures.csv'

    #逻辑处理函数
    def DelqFeatures(self, event, window, type):
        current = 12
        start = 12 - window + 1
        delq1 = [event[a] for a in ['Delq1_' + str(t) for t in range(current, start - 1, -1)]]
        delq2 = [event[a] for a in ['Delq2_' + str(t) for t in range(current, start - 1, -1)]]
        delq3 = [event[a] for a in ['Delq3_' + str(t) for t in range(current, start - 1, -1)]]
        if type == 'max delq':
            if max(delq3) == 1:
                return 3
            elif max(delq2) == 1:
                return 2
            elif max(delq1) == 1:
                return 1
            else:
                return 0
        if type in ['M0 times','M1 times', 'M2 times']:
            if type.find('M0')>-1:
                return sum(delq1)
            elif type.find('M1')>-1:
                return sum(delq2)
            else:
                return sum(delq3)

    # 考虑过去1个月，3个月，6个月，12个月
    def feature_extract(self, train_data):
        for t in [1, 3, 6, 12]:

            #过去t时间窗口内最大逾期状态
            derived_feature_name = 'maxDelqL' + str(t) + 'M'
            allFeatures.append(derived_feature_name)
            train_data[derived_feature_name] = train_data.apply(lambda x : self.DelqFeatures(x, t, 'max delq'),
                                                                          axis = 1)


            #过去t时间窗口内的，M0,M1,M2的次数
            allFeatures.append('M0FreqL' + str(t) + "M")
            train_data['M0FreqL' + str(t) + "M"] = train_data.apply(lambda x: self.DelqFeatures(x, t,'M0 times'),
                                                                              axis=1)

            allFeatures.append('M1FreqL' + str(t) + "M")
            train_data['M1FreqL' + str(t) + "M"] = train_data.apply(lambda x: self.DelqFeatures(x, t, 'M1 times'),
                                                                              axis=1)

            allFeatures.append('M2FreqL' + str(t) + "M")
            train_data['M2FreqL' + str(t) + "M"] = train_data.apply(lambda x: self.DelqFeatures(x, t, 'M2 times'),
                                                                              axis=1)



        self.delqFeature_Data = train_data[['CUST_ID'] + allFeatures]
        return  self.delqFeature_Data

    def save(self):

        self.delqFeature_Data.to_csv(ROOT_DIR + 'featureEngineering/' + self.result_file, index=None)

if __name__ == '__main__':
    impl = DelqFeatureExtractor()
    impl.feature_extract(train_data)
    impl.save()