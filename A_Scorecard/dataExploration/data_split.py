import pandas as pd

import pickle
from sklearn.model_selection import train_test_split
from  util.dataPreprocessing_function import *
from  settings import  *
# 数据预处理
# 1，读入数据
# 2，选择合适的建模样本
# 3，数据集划分成训练集和测试集

class dataSplit:
    def __init__(self, file):
           self.allData = pd.read_csv(file,header = 0, encoding = 'latin1')

    def sample_select(self,term,test_size):
        self.allData['term'] = self.allData['term'].apply(lambda x: int(x.replace(' months','')))
        # 处理标签：Fully Paid是正常用户:0；Charged Off是违约用户:1
        self.allData['y'] = self.allData['loan_status'].map(lambda x: int(x == 'Charged Off'))
        '''
        由于存在不同的贷款期限（term），申请评分卡模型评估的违约概率必须要在统一的期限中，且不宜太长，所以选取term＝36months的行本
        '''

        allData1 = self.allData.loc[self.allData.term == term]

        self.trainData1, self.testData1 = train_test_split(allData1,test_size=test_size)
        print('查看self.trainData1')
        print(self.trainData1.shape)   #(17457, 26)
        print('查看self.testData1')
        print(self.testData1.shape)   #(11638, 26)
        self.save()


    def save(self):
        with open(ROOT_DIR+'trainData.pkl','wb') as f:
            pickle.dump(self.trainData1, f)
        with open(ROOT_DIR+'testData.pkl','wb') as f1:
            pickle.dump(self.testData1, f1)





if __name__ == '__main__':
     datasplit = dataSplit(ROOT_DIR + 'application.csv')
     datasplit.sample_select(36, 0.4)

