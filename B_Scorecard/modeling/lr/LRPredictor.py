import  pandas as pd
import  statsmodels.api as sm
import pickle
import  matplotlib.pyplot as plt

from  sklearn.linear_model import  LogisticRegression
from  settings import  *
from  util.scorecard_functions import ks_auc_eval
from  util.scorecard_functions import Prob2Score

from  sklearn.externals import  joblib
from  featureEngineering.predict_data_generator import PredictionDataGenerator


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


    def predict(self, predict_df):

        X = self.dataGenerator.data_generate(predict_df)
        print('generate predict data:')
        print(X)

        result_df = pd.DataFrame()
        if self.platform == 'statsmodels':
            X['intercept'] = [1] * X.shape[0]
            result_df['pred'] = self.model.predict(X)
        else:
            X['intercept'] = [1] * X.shape[0]
            probas = self.model.predict_proba(X)[:, 1]
            result_df['pred'] = probas

        return  result_df



if __name__ == '__main__':

    #predictor = LogisticRegressionPredictor('statsmodels')
    predictor = LogisticRegressionPredictor('sklearn')
    test_df = pd.read_csv(ROOT_DIR + 'testData.csv')
    label = test_df['label']
    predict_df = test_df.drop(['CUST_ID', 'label'], axis = 1)

    pred = predictor.predict(predict_df)
    print('predict result:')
    print(pred)

    result_df = pd.DataFrame()
    result_df['pred'] = pred
    result_df['label'] = label

    ks_auc = ks_auc_eval(result_df)
    print(ks_auc)

    BasePoint, PDO = 500,50
    result_df['score'] = result_df['pred'].apply(lambda x: Prob2Score(x, BasePoint, PDO))
    print(result_df)
    plt.hist(result_df['score'],bins=100)
    plt.show()

