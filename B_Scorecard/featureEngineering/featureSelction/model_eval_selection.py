import  pandas as pd
import  pickle
import  numpy as np
from  sklearn.model_selection import  train_test_split
import  statsmodels.api as sm
import  matplotlib.pyplot as plt
from  featureEngineering.featureSelction.feature_importance import *
from  sklearn.metrics import  roc_auc_score
from  sklearn import  ensemble
from  settings import  *
from  util.scorecard_functions import *

train_data = pd.read_csv(ROOT_DIR + 'featureEngineering/train_WOE_data.csv')

#获取最优的alpha参数值
def L1_regularized(features):
    X = train_data[features]
    X['intercept'] = [1] * X.shape[0]
    y = train_data['label']
    for alpha in range(100,0,-1):
        logit = sm.Logit(y, X)
        l1_logit = sm.Logit.fit_regularized(logit,
                                            start_params=None,
                                            method='l1',
                                            alpha=alpha)
        pvalues = l1_logit.pvalues
        params = l1_logit.params
        if max(pvalues)>=0.1 or max(params)>0:
            break
    return  alpha


#计算单个变量特征的显著性P值
def get_single_pvalue(feature):
    X = train_data[[feature]]
    X['intercept'] = [1] * X.shape[0]
    y = train_data['label']
    logit = sm.Logit(y, X)
    logit_result = logit.fit()
    pvalues = logit_result.pvalues
    return  pvalues

#计算单个变量特征的回归系数值
def get_single_param(feature):
    X = train_data[feature]
    y = train_data['label']
    params = sm.Logit(y, X).fit().params
    return  params

def model_eval(features):
    print(features)
    X = train_data[features]
    X['intercept'] = [1] * X.shape[0]
    y = train_data['label']
    model = sm.Logit(y, X)
    logit_result = model.fit()
    pValues = logit_result.pvalues   #显著性水平

    print('p-values:')
    print(pValues)

    params = logit_result.params    #回归系数值
    print('params:')
    print(params)
    fit_result = pd.concat([params,pValues],axis=1)
    fit_result.columns = ['coefficient','p-value']
    fit_result = fit_result.sort_values(by = 'coefficient')

    return  (pValues, params)




#先假定模型可以容纳5个特征，再逐步增加特征个数，直到有特征的系数为正，或者p值超过0.1
def feature_select(multi_analysis_features):

    #1.先基于gbdt来获取特征重要性排序
    n = 5
    featuresImportance = get_fscore_gbdt(train_data, multi_analysis_features)
    featuresImportanceSorted = sorted(featuresImportance, key = lambda k : k[1], reverse = True)

    featuresSelected = [i[0] for i in featuresImportanceSorted[:n]]
    print(featuresSelected)

    #2.先对前5个得分最高的变量，观察变量系数情况
    model_eval(featuresSelected)

    k = n
    #3.依次加入剩余的特征，计算lr模型的变量系数正负性
    #一旦出现有特征的系数为正，说明该特征需要删除
    while (k < len(featuresImportanceSorted)):
        nextFeature = featuresImportanceSorted[k][0]
        featuresSelected = featuresSelected + [nextFeature]
        result = model_eval(featuresSelected)
        pValues = result[0]
        params = result[1]
        print('param for var: ', nextFeature, '   ', params[nextFeature])
        if max(params) < 0:
            k += 1
        else:
            featuresSelected.remove(nextFeature)
            k += 1
    print('final features list: ', featuresSelected)

    #4.再进一步考虑变量的显著性情况,观察P值是否会大于0.1
    pValues = model_eval(featuresSelected)[0]
    largePValuesFeature = pValues[pValues > 0.1].index

    print('large p-values feature list:')
    print(largePValuesFeature)

    #观察P值>0.1的特征，单独检验显著性
    #假如单个变量都是显著的，说明变量之间还存在着共线性
    for var in largePValuesFeature:
        pValues = get_single_pvalue(var)
        print('thie p-value of {0} is {1}'.format(var, str(pValues[var])))


    #5.基于L1约束 直到所有变量显著
    #在多重共线性场景下 L1约束 会倾向于把相对更“没用”的特征权值变成0
    alpha = L1_regularized(featuresSelected)
    bestAlpha = alpha
    y = train_data['label']
    X = train_data[featuresSelected]
    X['intercept'] = [1] * train_data.shape[0]
    l1_logit = sm.Logit.fit_regularized(sm.Logit(y, X),
                                        start_params=None,
                                        method='l1',
                                        alpha=bestAlpha)
    params = l1_logit.params
    params2 = params.to_dict()

    featuresInModel = [k for k, v in params2.items() if k!='intercept' and v < -0.0000001]

    # print('features In Model: ')
    # print(featuresInModel)

    for k, v in params2.items():
        print(k, '    ', v)

    with open(ROOT_DIR + 'featureEngineering/featuresInModel.pkl', 'wb') as f:
        pickle.dump(featuresInModel, f)




if __name__ == '__main__':
    with open(ROOT_DIR + 'featureEngineering/multi_analysis_feature_list.pkl', 'rb') as f:
        multi_analysis_features = pickle.load(f)
    #model_eval(multi_analysis_features)
    feature_select(multi_analysis_features)