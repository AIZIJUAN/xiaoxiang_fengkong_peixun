from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import  numpy as np
import  pandas as pd
import pickle

from  settings import  *

'''
orginal_data = pd.read_csv(ROOT_DIR + 'trainData.csv', header = None)
delqFeature_data = pd.read_csv(ROOT_DIR + 'featureEngineering/delqFeatures.csv')
paymentFeature_data = pd.read_csv(ROOT_DIR + 'featureEngineering/paymentFeatures.csv')
urateFeature_data = pd.read_csv(ROOT_DIR + 'featureEngineering/urateFeatures.csv')

derivedFeature_data = pd.merge(delqFeature_data, paymentFeature_data, on='CUST_ID', how='left')
derivedFeature_data = pd.merge(derivedFeature_data, urateFeature_data, on = 'CUST_ID', how = 'left')
'''


class  CorrelationAnalysisSelection:

    def IV_visualization(self):
        with open(ROOT_DIR +  'featureEngineering/IV_dict.pkl', 'rb') as f:
            IV_dict = pickle.load(f)
        #将变量IV值进行降序排列，方便后续挑选变量
        IV_dict_sorted = sorted(IV_dict.items(), key=lambda x: x[1], reverse=True)

        IV_values = [i[1] for i in IV_dict_sorted]
        IV_name = [i[0] for i in IV_dict_sorted]
        plt.title('feature IV')
        plt.bar(range(len(IV_values)),IV_values)

    def visualization(self, df, features):
        x = df[features]
        f, ax = plt.subplots(figsize=(10, 8))
        corr = x.corr()   #相关系数矩阵
        sns.heatmap(corr,
                    mask=np.zeros_like(corr, dtype=np.bool),
                    cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True,
                    ax=ax)

    '''
        单变量分析和多变量分析，均基于WOE编码后的值。
        （1）选择IV高于0.02的变量
        （2）比较两两线性相关性。如果相关系数的绝对值高于阈值，剔除IV较低的一个
    '''
    def woe_feature_analysis(self):
        with open(ROOT_DIR +  'featureEngineering/IV_dict.pkl', 'rb') as fa:
            IV_dict = pickle.load(fa)
        with open(ROOT_DIR +  'featureEngineering/WOE_dict.pkl', 'rb') as fb:
            WOE_dict = pickle.load(fb)
        train_data = pd.read_csv(ROOT_DIR + 'featureEngineering/train_WOE_data.csv',header = 0, encoding = 'latin1')

        #选取IV>0.02的变量
        high_IV = {k:v for k, v in IV_dict.items() if v >= 0.02}
        high_IV_sorted = sorted(high_IV.items(),key=lambda x:x[1],reverse=True)

        short_list = high_IV.keys()
        short_list_2 = []
        for var in short_list:
            newVar = var + '_WOE'
            train_data[newVar] = train_data[var].map(WOE_dict[var])
            short_list_2.append(newVar)

        #通过可视化来查看特征两两相关性情况
        self.visualization(train_data, short_list_2)

        #两两间的线性相关性检验
        #1，将候选变量按照IV进行降序排列
        #2，计算第i和第i+1的变量的线性相关系数
        #3，对于系数超过阈值的两个变量，剔除IV较低的一个
        deleted_index = []
        cnt_vars = len(high_IV_sorted)  #cnt_vars 表示IV较高的特征的个数
        for i in range(cnt_vars):
            if i in deleted_index:
                continue
            x1 = high_IV_sorted[i][0]+"_WOE"
            for j in range(cnt_vars):
                if i == j or j in deleted_index:
                    continue
                y1 = high_IV_sorted[j][0]+"_WOE"
                roh = np.corrcoef(train_data[x1],train_data[y1])[0,1]
                if abs(roh)>0.7:
                    x1_IV = high_IV_sorted[i][1]
                    y1_IV = high_IV_sorted[j][1]
                    if x1_IV > y1_IV:
                        deleted_index.append(j)
                    else:
                        deleted_index.append(i)

        multi_analysis_vars_1 = [high_IV_sorted[i][0]+"_WOE" for i in range(cnt_vars) if i not in deleted_index]


        '''
        2、多变量分析：VIF
        '''
        X = np.matrix(train_data[multi_analysis_vars_1])
        VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        max_VIF = max(VIF_list)
        print(max_VIF)
        # 最大的VIF是1.32267733123，因此这一步认为没有多重共线性
        multi_analysis = multi_analysis_vars_1

        print(multi_analysis)
        with open(ROOT_DIR + 'featureEngineering/multi_analysis_feature_list.pkl', 'wb') as f:
            pickle.dump(multi_analysis, f)





if __name__ == '__main__':
    analysis= CorrelationAnalysisSelection()
    analysis.woe_feature_analysis()


