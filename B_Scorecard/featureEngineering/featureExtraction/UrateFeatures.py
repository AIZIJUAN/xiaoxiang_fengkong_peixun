import  pandas as pd
import  numpy as np
from  settings import  *

train_data = pd.read_csv(ROOT_DIR + "trainData.csv")
allFeatures = []
class UrateFeaturesExtractor:
    def __init__(self):

        self.result_file = 'urateFeatures.csv'


    def UrateFeatures(self, event, window, type):
        current = 12
        start = 12 - window + 1
        monthlySpend = [event[a] for a in ['Spend_' + str(t) for t in range(current, start - 1, -1)]]
        limit = event['Loan_Amount']
        monthlyUrate = [x / limit for x in monthlySpend]
        if type == 'mean utilization rate':
            return np.mean(monthlyUrate)
        if type == 'max utilization rate':
            return max(monthlyUrate)
        if type == 'increase utilization rate':
            currentUrate = monthlyUrate[0 : -1]
            previousUrate = monthlyUrate[1:]
            compareUrate = [int(x[0]>x[1]) for x in zip(currentUrate,previousUrate)]
            return sum(compareUrate)


    """
    额度使用率特征
    """
    def feature_extract(self, train_data):
        for t in [1, 3, 6, 12]:
            # 1，过去t时间窗口内的最大月额度使用率
            allFeatures.append('maxUrateL' + str(t) + "M")
            train_data['maxUrateL' + str(t) + "M"] = train_data.apply(lambda x: self.UrateFeatures(x,t,'max utilization rate'),
                                                                                axis = 1)

            # 2，过去t时间窗口内的平均月额度使用率
            allFeatures.append('avgUrateL' + str(t) + "M")
            train_data['avgUrateL' + str(t) + "M"] = train_data.apply(lambda x: self.UrateFeatures(x, t, 'mean utilization rate'),
                                                                                axis=1)

            # 3，过去t时间窗口内，月额度使用率增加的月份。该变量要求t>1
            if t > 1:
                allFeatures.append('increaseUrateL' + str(t) + "M")
                train_data['increaseUrateL' + str(t) + "M"] = train_data.apply(lambda x: self.UrateFeatures(x, t, 'increase utilization rate'),
                                                                                         axis=1)


        self.urateFeature_data = train_data[['CUST_ID'] + allFeatures]
        return  self.urateFeature_data

    def save(self):
        self.urateFeature_data.to_csv(ROOT_DIR + 'featureEngineering/' + self.result_file, index= None)

if __name__ == "__main__":
    extractor = UrateFeaturesExtractor()
    extractor.feature_extract()
    extractor.save()