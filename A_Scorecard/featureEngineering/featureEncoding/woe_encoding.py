import  pandas as pd
import  numpy as np
import  pickle
from  settings import  *
from  util.scorecard_functions import  *

"""
对预处理后的特征进行分箱与WOE编码
"""
class WOEEncoding:

    def __init__(self, file):

        self.WOE_dict = {}
        self.IV_dict = {}
        self.train_data = pd.read_csv(file,header = 0, encoding = 'latin1')

        self.merge_bin_dict = {}  #存放需要合并的变量，以及合并方法
        self.br_encoding_dict = {}   #记录按照bad rate进行编码的变量，及编码方式
        self.continous_merged_dict = {}

    def feature_encoding_process(self):

        #对连续数值型变量分别进行卡方分箱与WOE编码
        self.compute_woe()
        #保存最终的WOE编码结果 以及相关特征分箱等信息，用于后续多变量分析 以及模型训练
        self.save()

    def compute_woe(self):
        more_value_features=[]
        less_value_features=[]
        # 第一步，检查类别型变量中，哪些变量取值超过5
        for var in cat_features:
            valueCounts = len(set(self.train_data[var]))
            print (valueCounts)
            if valueCounts > 5:
                more_value_features.append(var)  #取值超过5的变量，需要bad rate编码，再用卡方分箱法进行分箱
            else:
                less_value_features.append(var)

        var_bin_list = []   #由于某个取值没有好或者坏样本而需要合并的变量
        for col in less_value_features:
            binBadRate = BinBadRate(self.train_data, col, 'y')[0]
            if min(binBadRate.values()) == 0 :  #由于某个取值没有坏样本而进行合并
                print ('{} need to be combined due to 0 bad rate'.format(col))
                combine_bin = MergeBad0(self.train_data, col, 'y')
                self.merge_bin_dict[col] = combine_bin
                newVar = col + '_Bin'
                self.train_data[newVar] = self.train_data[col].map(combine_bin)
                var_bin_list.append(newVar)
            if max(binBadRate.values()) == 1:    #由于某个取值没有好样本而进行合并
                print ('{} need to be combined due to 0 good rate'.format(col))
                combine_bin = MergeBad0(self.train_data, col, 'y',direction = 'good')
                self.merge_bin_dict[col] = combine_bin
                newVar = col + '_Bin'
                self.train_data[newVar] = self.train_data[col].map(combine_bin)
                var_bin_list.append(newVar)

        #重新更新了less_value_features，只剩下不需要合并的变量
        less_value_features = [i for i in less_value_features if i + '_Bin' not in var_bin_list]

        # （ii）当取值>5时：用bad rate进行编码，放入连续型变量里,如：purpose
        #br_encoding_dict = {}   #记录按照bad rate进行编码的变量，及编码方式
        for col in more_value_features:
            br_encoding = BadRateEncoding(self.train_data, col, 'y')
            self.train_data[col+'_br_encoding'] = br_encoding['encoding']   #对每一列的值进行了映射
            self.br_encoding_dict[col] = br_encoding['bad_rate']     #只记录这一列的编码方式，不映射
            num_features.append(col+'_br_encoding')


        for col in num_features:
            print( "{} is in processing".format(col))
            if -1 not in set(self.train_data[col]):   #－1会当成特殊值处理。如果没有－1，则所有取值都参与分箱
                max_interval = 5   #分箱后的最多的箱数
                cutOff = ChiMerge(self.train_data, col, 'y', max_interval=max_interval,special_attribute=[],minBinPcnt=0)
                self.train_data[col+'_Bin'] = self.train_data[col].map(lambda x: AssignBin(x, cutOff,special_attribute=[]))
                monotone = BadRateMonotone(self.train_data, col+'_Bin', 'y')   # 检验分箱后的单调性是否满足
                while(not monotone):
                    # 检验分箱后的单调性是否满足。如果不满足，则缩减分箱的个数。
                    max_interval -= 1
                    cutOff = ChiMerge(self.train_data, col, 'y', max_interval=max_interval, special_attribute=[],
                                                  minBinPcnt=0)
                    self.train_data[col + '_Bin'] = self.train_data[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[]))
                    if max_interval == 2:
                        # 当分箱数为2时，必然单调
                        break
                    monotone = BadRateMonotone(self.train_data, col + '_Bin', 'y')
                newVar = col + '_Bin'
                self.train_data[newVar] = self.train_data[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[]))
                var_bin_list.append(newVar)
            else:
                max_interval = 5
                # 如果有－1，则除去－1后，其他取值参与分箱
                cutOff = ChiMerge(self.train_data, col, 'y', max_interval=max_interval, special_attribute=[-1],
                                              minBinPcnt=0)
                self.train_data[col + '_Bin'] = self.train_data[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
                monotone = BadRateMonotone(self.train_data, col + '_Bin', 'y',['Bin -1'])
                while (not monotone):
                    max_interval -= 1
                    # 如果有－1，－1的bad rate不参与单调性检验
                    cutOff = ChiMerge(self.train_data, col, 'y', max_interval=max_interval, special_attribute=[-1],
                                                  minBinPcnt=0)
                    self.train_data[col + '_Bin'] = self.train_data[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
                    if max_interval == 3:
                        # 当分箱数为3-1=2时，必然单调
                        break
                    monotone = BadRateMonotone(self.train_data, col + '_Bin', 'y',['Bin -1'])
                newVar = col + '_Bin'
                self.train_data[newVar] = self.train_data[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
                var_bin_list.append(newVar)
            self.continous_merged_dict[col] = cutOff
        print('查看self.train_data')
        print(self.train_data)
        '''WOE编码  '''
        # 分箱后的变量进行编码，包括：
        # 1，初始取值个数小于5，且不需要合并的类别型变量。存放在less_value_features中
        # 2，初始取值个数小于5，需要合并的类别型变量。合并后新的变量存放在var_bin_list中
        # 3，初始取值个数超过5，需要合并的类别型变量。合并后新的变量存放在var_bin_list中
        # 4，连续变量。分箱后新的变量存放在var_bin_list中
        all_var = var_bin_list  + less_value_features
        for var in all_var:
            new_var = var + '_WOE'
            woe_iv = CalcWOE(self.train_data, var, 'y')
            self.WOE_dict[new_var] = woe_iv['WOE']
            self.IV_dict[new_var] = woe_iv['IV']
            self.train_data[new_var] = self.train_data[var].map(lambda x : self.WOE_dict[new_var][x])
            print(self.WOE_dict.get(new_var))

    def save(self):

        #将所有经过WOE编码的新特征及相关WOE,IV值保存在本地
        with open(ROOT_DIR + 'featureEngineering/WOE_dict.pkl', 'wb') as fa:
            pickle.dump(self.WOE_dict, fa)

        with open(ROOT_DIR + 'featureEngineering/IV_dict.pkl', 'wb') as fb:
            pickle.dump(self.IV_dict, fb)
        #print(self.train_data.columns)
        self.train_data.to_csv(ROOT_DIR + 'featureEngineering/train_WOE_data.csv', index=None)  #包含需要计算WOE的变量和目标变量

        with open(ROOT_DIR + 'featureEngineering/merge_bin_dict.pkl', 'wb') as f1:
            pickle.dump(self.merge_bin_dict, f1)

        with open(ROOT_DIR + 'featureEngineering/br_encoding_dict.pkl', 'wb') as f2:
            pickle.dump(self.br_encoding_dict, f2)

        with open(ROOT_DIR + 'featureEngineering/continous_merged_dict.pkl', 'wb') as f3:
            pickle.dump(self.continous_merged_dict, f3)

if __name__ == '__main__':
    woeEncoding = WOEEncoding(ROOT_DIR + 'dataExploration/dealed_trainData.csv')
    woeEncoding.feature_encoding_process()