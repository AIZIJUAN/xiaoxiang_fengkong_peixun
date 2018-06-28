import  pandas as pd
import  statsmodels.api as sm
import pickle
import  matplotlib.pyplot as plt

from  sklearn.linear_model import  LogisticRegression
from  settings import  *
from  util.scorecard_functions import ks_auc_eval
from  util.scorecard_functions import *
from  sklearn.externals import  joblib
from  featureEngineering.predict_data_generator import PredictionDataGenerator
from  sklearn.metrics import  roc_auc_score

class LogisticRegressionPredictor:

    def __init__(self, platform):
        self.platform = platform
        self.dataGenerator = PredictionDataGenerator()
        self.model = self.load_model(platform)
        print('load local model file, platfrom:', platform)

    def load_model(self, platform):
        model_name = 'LR-Model-' + platform + '.m'
        model = joblib.load(model_name)
        return  model


    def ks_auc_eval(self, result_df):

        ks = KS(result_df, 'pred', 'y')
        auc = roc_auc_score(result_df['y'], result_df['pred'])  #AUC = 0.73
        #{'AUC': 0.83644931044825688, 'KS': 0.59816049348012412}
        result = {}
        result['ks'] = ks
        result['auc'] = auc
        return  result

    def predict(self, test_df):

        y_test = test_df['y']
        print('查看y_test')
        print(y_test)
        X_test = test_df.drop(['y'], axis = 1)
        del test_df

        X = self.dataGenerator.data_generate(X_test)
        #print('generate predict data:')
        #print(X)

        result_df = pd.DataFrame()
        if self.platform == 'statsmodels':
            X['intercept'] = [1] * X.shape[0]
            result_df['pred'] = self.model.predict(X)
            result_df['y'] = y_test
            result_df['y'] =result_df['y'].astype(int)
        else:
            X['intercept'] = [1] * X.shape[0]
            probas = self.model.predict_proba(X)[:, 1]
            result_df['pred'] = probas
            result_df['y'] = y_test
            result_df['y'] =result_df['y'].astype(int)
        print('查看y_test')
        print(y_test)
        print('查看result_df_y')
        print(result_df['y'])
        return  result_df



if __name__ == '__main__':

    #predictor = LogisticRegressionPredictor('statsmodels')
    predictor = LogisticRegressionPredictor('sklearn')
    test_df = pd.read_csv(ROOT_DIR + 'dataExploration/testData.csv',header = 0, encoding = 'latin1')

    result_df = predictor.predict(test_df)
    #print('predict result:')
    #print(pred)

    ks_auc = predictor.ks_auc_eval(result_df)

    print(ks_auc)

    BasePoint, PDO = 500,50
    result_df['score'] = result_df['pred'].apply(lambda x: Prob2Score(x, BasePoint, PDO))
    print(result_df)
    plt.hist(result_df['score'],bins=100)
    plt.show()

