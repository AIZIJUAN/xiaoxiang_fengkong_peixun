#conding = utf-8

import  pandas as pd

target = "y"

from  example.scorecard_functions_V3 import *

train_data_file = "D:/conf_test/A_Scorecard/application.csv"



def BadRateEncodingTest():

    df = pd.read_csv(train_data_file, encoding = 'latin1')
    # 处理标签：Fully Paid是正常用户；Charged Off是违约用户
    df['y'] = df['loan_status'].map(lambda x: int(x == 'Charged Off'))

    col = "home_ownership"

    regroup = BinBadRate(df, col, target, grantRateIndicator=0)[1]

    print("regroup:")
    print(regroup)

    temp_regroup = regroup[[col,'bad_rate']].set_index([col])
    print("temp group:")
    print(temp_regroup)
    br_dict = regroup[[col,'bad_rate']].set_index([col]).to_dict(orient='index')

    print("br_dict:")
    print(br_dict)

    for k, v in br_dict.items():
        print(k)
        print(v)
        br_dict[k] = v['bad_rate']
    badRateEnconding = df[col].map(lambda x: br_dict[x])


if __name__ == "__main__":
    BadRateEncodingTest()