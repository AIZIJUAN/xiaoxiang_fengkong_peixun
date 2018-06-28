import  pandas as pd

from  settings import  *
from featureEngineering.featureExtraction.DelqFeatures import *
from featureEngineering.featureExtraction.PaymentFeatures import *
from featureEngineering.featureExtraction.UrateFeatures import *

class ExtractionMain:

    def initial(self):
        self.delqExtractor = DelqFeatureExtractor()
        self.paymentExtractor = PaymentFeaturesExtractor()
        self.urateExtractor = UrateFeaturesExtractor()

        self.train_data = pd.read_csv(ROOT_DIR + 'trainData.csv')


    def feature_extract(self):

        self.initial()

        delqFeature = self.delqExtractor.feature_extract(self.train_data)
        paymentFeature = self.paymentExtractor.feature_extract(self.train_data)
        urateFeature = self.urateExtractor.feature_extract(self.train_data)

        feature_data  = pd.merge(delqFeature, paymentFeature, on='CUST_ID', how='left')
        feature_data = pd.merge(feature_data, urateFeature, on = 'CUST_ID', how = 'left')

        feature_data = pd.merge(feature_data, self.train_data[['CUST_ID', 'label']], on = 'CUST_ID', how='left')

        self.save(feature_data)


    def save(self, feature_data_df):
        feature_data_df.to_csv(ROOT_DIR + 'featureEngineering/train_derived_feature_data.csv', index = None)

    #其中某一类特征重新计算后，update整体训练特征集
    def update_feature(self):

        train_label_data = pd.read_csv(ROOT_DIR + 'trainData.csv')[['CUST_ID', 'label']]

        delqFeature_df = pd.read_csv(ROOT_DIR +  'featureEngineering/delqFeatures.csv')
        paymentFeature_df = pd.read_csv(ROOT_DIR +  'featureEngineering/paymentFeatures.csv')
        urateFeature_df = pd.read_csv(ROOT_DIR +  'featureEngineering/urateFeatures.csv')

        feature_data  = pd.merge(delqFeature_df, paymentFeature_df, on='CUST_ID', how='left')
        feature_data = pd.merge(feature_data, urateFeature_df, on = 'CUST_ID', how = 'left')

        feature_data = pd.merge(feature_data, train_label_data, on = 'CUST_ID', how='left')

        self.save(feature_data)


if __name__ == '__main__':
    extractionMain = ExtractionMain()
    #feature_data_df = extractionMain.feature_extract()
    extractionMain.update_feature()
