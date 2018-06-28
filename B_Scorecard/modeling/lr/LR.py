#coding=utf-8

import  pickle
import  pandas as pd
from  sklearn.metrics import  roc_auc_score
from  sklearn.linear_model import  LogisticRegression
import  statsmodels.api as sm
from  util.scorecard_functions import *
from settings import  *

from sklearn.externals import joblib



class LogisticRegressionRunner:
    def __init__(self):
        with open(ROOT_DIR + 'featureEngineering/featuresInModel.pkl', 'rb') as f:
            self.featuresInModel = pickle.load(f)

        train_data = pd.read_csv(ROOT_DIR + 'featureEngineering/train_WOE_data.csv')[self.featuresInModel + ['label']]

        self.y_train = train_data['label']
        self.X_train = train_data.drop(['label'], axis = 1)
        del train_data

    def ks_auc_eval(self, result_df):

        ks = KS(result_df, 'pred', 'label')
        auc = roc_auc_score(result_df['label'], result_df['pred'])  #AUC = 0.73
        #{'AUC': 0.83644931044825688, 'KS': 0.59816049348012412}
        result = {}
        result['ks'] = ks
        result['auc'] = auc
        return  result



    def train(self, platform):
        X = self.X_train
        X['intercept'] = [1] * X.shape[0]
        y = self.y_train


        if platform == 'statsmodels':
            logit = sm.Logit(y, X)
            logit_result = logit.fit()

            result_df = pd.DataFrame()
            result_df['label'] = self.y_train
            result_df['pred'] = logit_result.predict(X)

            print(result_df)

            self.save_model(logit_result, 'LR-Model-statsmodels.m')

        elif platform == 'sklearn':
            clf = LogisticRegression(penalty='l1',  #正则化策略
                                     dual=False,
                                     tol=0.000001,   #迭代收敛阈值
                                     C=2.0,   #惩罚系数倒数
                                     class_weight='balanced',
                                     random_state=1024,
                                     solver='liblinear',  #最优化问题算法
                                     max_iter=10000,
                                     verbose=0,
                                     n_jobs=-1)

            logit_result = clf.fit(X, y)
            probas = logit_result.predict_proba(X)[:, 1]

            result_df = pd.DataFrame()
            result_df['label'] = self.y_train
            result_df['pred'] = probas

            print(result_df)

            self.save_model(logit_result, 'LR-Model-sklearn.m')

        else:
            raise  RuntimeError('option for platform is invalid')


        return  result_df

    def save_model(self, model, fileName):
        joblib.dump(model, fileName)

if __name__ == '__main__':
    lr = LogisticRegressionRunner()
    result_df = lr.train(platform='sklearn')
    ks_auc = lr.ks_auc_eval(result_df)
    print(ks_auc)
