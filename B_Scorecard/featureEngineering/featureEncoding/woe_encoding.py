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
        self.categoricalFeatures = []
        self.numericalFeatures = []
        self.WOE_IV_dict = {}
        self.train_data = pd.read_csv(file)

        self.toRemove_list = []

        self.bin_dict = []

    def feature_encoding_process(self):
        #按照值分布，将特征分为类别型变量和连续性变量
        featureList = [x for x in self.train_data.columns if x not in ['CUST_ID', 'label']]
        for f in featureList:
            if len(set(self.train_data[f])) > NUM_BINS:
                self.numericalFeatures.append(f)
            else:
                self.categoricalFeatures.append(f)



        #类别型变量与连续数值型变量分别进行卡方分箱与WOE编码
        self.categorical_feature_encoding()
        self.numerical_feature_encoding()

        #保存最终的WOE编码结果 以及相关特征分箱等信息，用于后续多变量分析 以及模型训练
        self.save()


    def not_monotone_feature_process(self, not_monotone_list):
        print(not_monotone_list)
        '''
        for var in not_monotone_list:
            print(self.train_data.groupby(var)['label'].mean())
            print(self.train_data.groupby(var)['label'].count())
        '''

        self.train_data['M1FreqL3M_Bin'] = self.train_data['M1FreqL3M'].apply(lambda x: int(x >= 1))
        print(self.train_data.groupby('M1FreqL3M_Bin')['label'].mean())

        self.train_data['M2FreqL3M_Bin'] = self.train_data['M2FreqL3M'].apply(lambda x : int(x >= 1))
        print(self.train_data.groupby('M2FreqL3M_Bin')['label'].mean())

        #计算WOE值
        self.compute_woe('M1FreqL3M_Bin')
        self.compute_woe('M2FreqL3M_Bin')


    def compute_woe(self, var):
        new_var = var + '_WOE'
        self.WOE_IV_dict[new_var] = CalcWOE(self.train_data, var, 'label')
        self.train_data[new_var] = self.train_data[var].map(lambda x : self.WOE_IV_dict[new_var]['WOE'][x])
        print(self.WOE_IV_dict.get(new_var))


    def categorical_feature_encoding(self):

        not_monotone = []

        #1.先对类别变量中badrate不单调的特征进行合并并且计算WOE值
        for var in self.categoricalFeatures:
            if not BadRateMonotone(self.train_data, var, target='label'):
                not_monotone.append(var)

        self.not_monotone_feature_process(not_monotone)

        #2.针对其他单调的类别型变量，检查是否有一箱的占比低于5%。 如果有，将该变量进行合并
        #根据是否存在有小于5%的bin，将特征分为small_bin_var与large_bin_var两个集合
        #依次对small_bin_var与large_bin_var两个集合中的特征进行分析处理
        small_bin_var = []
        large_bin_var = []
        N = self.train_data.shape[0]
        for var in self.categoricalFeatures:
            if var not in not_monotone:
                total = self.train_data.groupby([var])[var].count()
                pcnt = total * 1.0 / N
                if min(pcnt) < 0.05:
                    small_bin_var.append({var : pcnt.to_dict()})
                else:
                    large_bin_var.append(var)

        #2.1针对 samll_bin_var中的变量进行处理
        for i in small_bin_var:
            print(i)

        '''
        正常的变量：
        {'maxDelq1M': {0: 0.60379372931421049, 1: 0.31880138083205806, 2: 0.069183956724438597, 3: 0.0082209331292928574}}

        {'maxDelq3M': {0: 0.22637816292394747, 1: 0.57005587387451506, 2: 0.18068258656891703, 3: 0.022883376632620377}}

        {'maxDelq6M': {0: 0.057226235809103528, 1: 0.58489625965336844, 2: 0.31285810882949572, 3: 0.045019395708032317}}

        需要被删除的无效变量：
        {'M2FreqL1M': {0: 0.99177906687070716, 1: 0.0082209331292928574}}
        {'M2FreqL6M': {0: 0.95498060429196774, 1: 0.04003701199330937, 2: 0.0045909107085661408, 3: 0.00032029609594647497,
                        4: 7.1176910210327775e-05}}

        {'M2FreqL12M': {0: 0.92334246770347694, 1: 0.066514822591551295, 2: 0.0092174098722374465, 3: 0.00081853446741876937,
                        4: 0.00010676536531549166}}
        '''
        #M2FreqL1M, M2FreqL6M, M2FreqL12M三个特征的分箱分布，尤其特别不平衡，可能会对模型预测带来负面效果，因此删除
        self.toRemove_list.append('M2FreqL1M')
        self.toRemove_list.append('M2FreqL6M')
        self.toRemove_list.append('M2FreqL12M')

        #maxDelq1M， maxDelq3M， maxDelq6M 对于这三个正常的变量，进行分箱合并，计算WOE值
        self.train_data['maxDelqL1M_Bin'] = self.train_data['maxDelqL1M'].apply(lambda x: MergeByCondition(x,['==0','==1','>=2']))
        self.train_data['maxDelqL3M_Bin'] = self.train_data['maxDelqL3M'].apply(lambda x: MergeByCondition(x,['==0','==1','>=2']))
        self.train_data['maxDelqL6M_Bin'] = self.train_data['maxDelqL6M'].apply(lambda x: MergeByCondition(x,['==0','==1','>=2']))

        self.compute_woe('maxDelqL1M_Bin')
        self.compute_woe('maxDelqL3M_Bin')
        self.compute_woe('maxDelqL6M_Bin')

        #2.2针对large_bin_var中的变量进行处理
        #对于不需要分箱合并，原始特征的badrate就已经单调的变量直接计算WOE和IV值
        for var in large_bin_var:
            self.compute_woe(var)

    '''
    对于数值型变量，需要先分箱，再计算WOE、IV
    分箱的结果需要满足：
    1:箱数不超过5
    2:bad rate单调
    3:每箱占比不低于5%
    '''

    def numerical_feature_encoding(self):

        for var in self.numericalFeatures:
            max_bins = NUM_BINS
            print(var)
            #先进行卡方分箱，将特征数据离散化，并且按照卡方合并的原理，将原始特征的bin数控制在num_bin范围以内
            bin = ChiMerge(self.train_data, var, 'label', max_interval=max_bins, minBinPcnt= 0.05)
            print('chiMerge bin: ', bin)
            new_var = var + '_Bin'
            self.train_data[new_var] = self.train_data[var].apply(lambda x : AssignBin(x, bin))

            while not BadRateMonotone(self.train_data, new_var, 'label'):
                print('not monotone, ChiMerge to make badrate monotone  for var: ', var)
                max_bins -= 1  #降低分箱数，进一步合并，再判断badrate是否能满足单调
                print('max bin:', max_bins)
                bin = ChiMerge(self.train_data, var, 'label', max_interval=max_bins, minBinPcnt= 0.05)
                self.train_data[new_var] = self.train_data[var].apply(lambda x: AssignBin(x, bin))

            #满足单调性后计算WOE值
            self.compute_woe(new_var)
            self.bin_dict.append({var: bin})

    def save(self):

        #将所有经过WOE编码的新特征及相关WOE,IV值保存在本地
        with open(ROOT_DIR + 'featureEngineering/WOE_IV_dict.pkl', 'wb') as f:
            print(self.WOE_IV_dict)
            pickle.dump(self.WOE_IV_dict, f)

        #print(self.train_data.columns)
        self.train_data.to_csv(ROOT_DIR + 'featureEngineering/train_WOE_data.csv', index=None)

        with open(ROOT_DIR + 'featureEngineering/numericalFeatures.pkl', 'wb') as f1:
            pickle.dump(self.numericalFeatures, f1)

        with open(ROOT_DIR + 'featureEngineering/categoricalFeatures.pkl', 'wb') as f2:
            pickle.dump(self.categoricalFeatures, f2)

        with open(ROOT_DIR + 'featureEngineering/bin_dict.pkl', 'wb') as f3:
            pickle.dump(self.bin_dict, f3)

if __name__ == '__main__':
    woeEncoding = WOEEncoding(ROOT_DIR + 'featureEngineering/train_derived_feature_data.csv')
    woeEncoding.feature_encoding_process()