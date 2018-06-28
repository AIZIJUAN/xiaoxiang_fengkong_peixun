import  pandas as pd
import  pickle

from  settings import  *

from  util.scorecard_functions import *
from  util.dataPreprocessing_function import *

class PredictionDataGenerator:

    def __init__(self):
        with open(ROOT_DIR + 'featureEngineering/merge_bin_dict.pkl', 'rb') as f2:
            self.merge_bin_dict = pickle.load(f2)
            print(self.merge_bin_dict)

        with open(ROOT_DIR + 'featureEngineering/br_encoding_dict.pkl', 'rb') as f3:
            self.br_encoding_dict = pickle.load(f3)

        with open(ROOT_DIR + 'featureEngineering/continous_merged_dict.pkl', 'rb') as f4:
            self.continous_merged_dict = pickle.load(f4)


        with open(ROOT_DIR + 'featureEngineering/featuresInModel.pkl', 'rb') as f5:
            self.featuresInModel = pickle.load(f5)

        with open(ROOT_DIR + 'featureEngineering/WOE_dict.pkl', 'rb') as f6:
            self.WOE_dict = pickle.load(f6)



    def data_generate(self, predict_df):

        # 将带％的百分比变为浮点数
        predict_df['int_rate_clean'] = predict_df['int_rate'].map(lambda x: float(x.replace('%',''))/100)

        # 将工作年限进行转化，否则影响排序
        predict_df['emp_length_clean'] = predict_df['emp_length'].map(CareerYear)

        # 将desc的缺失作为一种状态，非缺失作为另一种状态
        predict_df['desc_clean'] = predict_df['desc'].map(DescExisting)

        # 处理日期。earliest_cr_line的格式不统一，需要统一格式且转换成python的日期
        predict_df['app_date_clean'] = predict_df['issue_d'].map(lambda x: ConvertDateStr(x))
        predict_df['earliest_cr_line_clean'] = predict_df['earliest_cr_line'].map(lambda x: ConvertDateStr(x))

        # 处理mths_since_last_delinq。注意原始值中有0，所以用－1代替缺失
        predict_df['mths_since_last_delinq_clean'] = predict_df['mths_since_last_delinq'].map(lambda x:MakeupMissing(x))

        predict_df['mths_since_last_record_clean'] = predict_df['mths_since_last_record'].map(lambda x:MakeupMissing(x))

        predict_df['pub_rec_bankruptcies_clean'] = predict_df['pub_rec_bankruptcies'].map(lambda x:MakeupMissing(x))

        '''
        第二步：变量衍生
        '''
        # 考虑申请额度与收入的占比
        predict_df['limit_income'] = predict_df.apply(lambda x: x.loan_amnt / x.annual_inc, axis = 1)

        # 考虑earliest_cr_line到申请日期的跨度，以月份记
        predict_df['earliest_cr_to_app'] = predict_df.apply(lambda x: MonthGap(x.earliest_cr_line_clean,x.app_date_clean), axis = 1)

        '''
        第三步：分箱并代入WOE值
        '''
        for var in self.featuresInModel:
            var1 = var.replace('_Bin_WOE','')

            # 有些取值个数少、但是需要合并的变量
            if var1 in self.merge_bin_dict.keys():
                print ("{} need to be regrouped".format(var1))
                predict_df[var1 + '_Bin'] = predict_df[var1].map(self.merge_bin_dict[var1])

            # 有些变量需要用bad rate进行编码
            if var1.find('_br_encoding')>-1:
                var2 =var1.replace('_br_encoding','')
                print( "{} need to be encoded by bad rate".format(var2))
                predict_df[var1] = predict_df[var2].map(self.br_encoding_dict[var2])
                #需要注意的是，有可能在测试样中某些值没有出现在训练样本中，从而无法得出对应的bad rate是多少。故可以用最坏（即最大）的bad rate进行编码
                max_br = max(predict_df[var1])
                predict_df[var1] = predict_df[var1].map(lambda x: ModifyDf(x, max_br))


            #上述处理后，需要加上连续型变量一起进行分箱
            if -1 not in set(predict_df[var1]):
                predict_df[var1+'_Bin'] = predict_df[var1].map(lambda x: AssignBin(x, self.continous_merged_dict[var1]))
            else:
                predict_df[var1 + '_Bin'] = predict_df[var1].map(lambda x: AssignBin(x, self.continous_merged_dict[var1],[-1]))

            #WOE编码
            var3 = var.replace('_WOE','')
            predict_df[var] = predict_df[var3].map(self.WOE_dict[var])


        return  predict_df[self.featuresInModel]