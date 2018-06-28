import pandas as pd

import pickle
from sklearn.model_selection import train_test_split
from  util.dataPreprocessing_function import *
from  settings import  *

'''
第一步：数据预处理，包括
（1）数据清洗
（2）格式转换
（3）确实值填补
'''

'''
数据与描述
int_rate:利率
emp_length：工作年限
desc：申请时的描述
issue_d：申请提交日期
earliest_cr_line：
mths_since_last_delinq：上次逾期距今月份数
mths_since_last_record：上次登记公众纪录距今的月份数
pub_rec_bankruptcies：公众破产记录数

衍生变量需要以下变量
loan_amnt：申请额度
annual_inc：年收入
'''

class PreprocessingDataGenerator:

    def __init__(self):
        with open(ROOT_DIR + 'trainData.pkl', 'rb') as f1:
            self.trainData = pickle.load(f1)
        with open(ROOT_DIR + 'testData.pkl', 'rb') as f2:
            self.testData = pickle.load(f2)
        self.result_file1 = 'dealed_trainData.csv'
        self.result_file2 = 'testData.csv'

    def preprocess_traindata(self):

        # 将带％的百分比变为浮点数
        #将内容增加到副本上面去前，需要将副本拷贝一下，否则会弹出A value is trying to be set on a copy of a slice from a DataFrame.
        trainData=self.trainData
        trainData['int_rate_clean'] = trainData['int_rate'].map(lambda x: float(x.replace('%',''))/100)

        # 将工作年限进行转化，否则影响排序
        trainData['emp_length_clean'] = trainData['emp_length'].map(CareerYear)

        # 将desc的缺失作为一种状态，非缺失作为另一种状态
        trainData['desc_clean'] = trainData['desc'].map(DescExisting)

        # 处理日期。earliest_cr_line的格式不统一，需要统一格式且转换成python的日期
        trainData['app_date_clean'] = trainData['issue_d'].map(lambda x: ConvertDateStr(x))
        trainData['earliest_cr_line_clean'] = trainData['earliest_cr_line'].map(lambda x: ConvertDateStr(x))

        # 处理mths_since_last_delinq。注意原始值中有0，所以用－1代替缺失
        # 上次逾期距今月份数
        trainData['mths_since_last_delinq_clean'] = trainData['mths_since_last_delinq'].map(lambda x:MakeupMissing(x))
        #上次登记公众纪录距今月份数
        trainData['mths_since_last_record_clean'] = trainData['mths_since_last_record'].map(lambda x:MakeupMissing(x))
        #公众破产记录数
        trainData['pub_rec_bankruptcies_clean'] = trainData['pub_rec_bankruptcies'].map(lambda x:MakeupMissing(x))

        '''
        第二步：变量衍生
        '''
        # 考虑申请额度与收入的占比
        trainData['limit_income'] = trainData.apply(lambda x: x.loan_amnt / x.annual_inc, axis = 1)

        # 考虑earliest_cr_line到申请日期的跨度，以月份记
        trainData['earliest_cr_to_app'] = trainData.apply(lambda x: MonthGap(x.earliest_cr_line_clean,x.app_date_clean), axis = 1)
        print('查看trainData')
       # print(trainData)    #对上述10个特征做了处理并生成新特征，目前有36个特征
        print(trainData['y'])
        self.save_train(trainData)

    def save_train(self, data_df):
        data_df.to_csv(ROOT_DIR + 'dataExploration/'+self.result_file1, index = None)
        self.testData.to_csv(ROOT_DIR + 'dataExploration/'+self.result_file2, index = None)

if __name__ == '__main__':
     impl = PreprocessingDataGenerator()
     impl.preprocess_traindata()