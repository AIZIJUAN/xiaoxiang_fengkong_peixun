#coding=utf-8
import  pickle
import  pandas as pd
import  xgboost as xgb
from  sklearn.ensemble import  GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import  operator

from  settings import  *


#train_data = pd.read_csv(ROOT_DIR + 'featureEngineering/train_WOE_data.csv')

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def get_fscore_xgb(train_data, features):
    X = train_data[features]
    y = train_data['y']

    dtrain = xgb.DMatrix(X, label=y)

    clf = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=20
    )

    create_feature_map(features)
    importance = clf.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key = operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'score'])
    print(df)



def get_fscore_gbdt(train_data, features):
    X = train_data[features]
    y = train_data['y']

    gbClassifier = GradientBoostingClassifier()
    model = gbClassifier.fit(X, y)
    importance = model.feature_importances_.tolist()

    featuresImportance = zip(features, importance)
    featuresImportanceSorted = sorted(featuresImportance, key = lambda k : k[1], reverse = True)
    print(featuresImportanceSorted)

    return  featuresImportanceSorted



# 用随机森林法估计变量重要性#
def get_fscore_rfc(self, features):
     print(self.train_data)

     X = self.train_data[features]
     y = self.train_data['y']

     RFC = RandomForestClassifier()
     RFC_Model = RFC.fit(X,y)
     features_rfc = X.columns
     featuresImportance = {features_rfc[i]:RFC_Model.feature_importances_[i] for i in range(len(features_rfc))}
     featuresImportanceSorted = sorted(featuresImportance.items(),key=lambda x: x[1], reverse=True)
     print(featuresImportanceSorted)

     return featuresImportanceSorted


if __name__ == '__main__':
    with open(ROOT_DIR + 'featureEngineering/multi_analysis_feature_list.pkl', 'rb') as f:
        features = pickle.load(f)

    #get_fscore_rfc(features)

