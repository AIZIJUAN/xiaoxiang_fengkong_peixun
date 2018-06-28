import  pandas as pd

from  settings import *

train_data = pd.read_csv(ROOT_DIR + 'trainData.csv')

allFeatures = []
class PaymentFeaturesExtractor:
    def __init__(self):

        self.result_file = 'paymentFeatures.csv'

    #OS: outstanding,表示尚未偿还的贷款量
    def PaymentFeatures(self, event, window, type):
        current = 12
        start = 12 - window + 1
        currentPayment = [event[a] for a in ['Payment_' + str(t) for t in range(current, start - 1, -1)]]
        previousOS = [event[a] for a in ['OS_' + str(t) for t in range(current - 1, start - 2, -1)]]
        monthlyPayRatio = []
        for Pay_OS in zip(currentPayment,previousOS):
            if Pay_OS[1]>0:
                payRatio = Pay_OS[0]*1.0 / Pay_OS[1]
                monthlyPayRatio.append(payRatio)
            else:
                monthlyPayRatio.append(1)
        if type == 'min payment ratio':
            return min(monthlyPayRatio)
        if type == 'max payment ratio':
            return max(monthlyPayRatio)
        if type == 'mean payment ratio':
            total_payment = sum(currentPayment)
            total_OS = sum(previousOS)
            if total_OS > 0:
                return total_payment / total_OS
            else:
                return 1

    '''
    还款类型特征提取
    '''
    def feature_extract(self, train_data):
        for t in [1, 3, 6, 12]:
            # 1，过去t时间窗口内的最大月还款率
            allFeatures.append('maxPayL' + str(t) + "M")
            train_data['maxPayL' + str(t) + "M"] = train_data.apply(lambda x: self.PaymentFeatures(x, t, 'max payment ratio'),
                                                                                axis=1)

            # 2，过去t时间窗口内的最小月还款率
            allFeatures.append('minPayL' + str(t) + "M")
            train_data['minPayL' + str(t) + "M"] = train_data.apply(lambda x: self.PaymentFeatures(x, t, 'min payment ratio'),
                                                                    axis=1)

            # 3，过去t时间窗口内的平均月还款率
            allFeatures.append('avgPayL' + str(t) + "M")
            train_data['avgPayL' + str(t) + "M"] = train_data.apply(lambda x: self.PaymentFeatures(x, t, 'mean payment ratio'),
                                                                    axis=1)

        self.paymentFeature_data = train_data[['CUST_ID'] + allFeatures]
        return  self.paymentFeature_data

    def save(self):
        self.paymentFeature_data.to_csv(ROOT_DIR + 'featureEngineering/' + self.result_file, index= None)

if __name__ == "__main__":
    extractor = PaymentFeaturesExtractor()
    extractor.feature_extract()
    extractor.save()
