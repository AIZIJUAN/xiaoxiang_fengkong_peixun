# -*- coding: utf-8 -*

class _const:
    class ConstError(TypeError): pass
    class ConstCaseError(ConstError): pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("can't change const %s" % name)
        if not name.isupper():
            raise self.ConstCaseError("const name %s is not all uppercase" % name)
        self.__dict__[name] = value

const = _const()

#原始数据保存路径
const.FOLD_OF_DATA = 'C:/Users/daill/PycharmProjects/C_Scorecard/data/'
#原始数据csv文件名称
const.SRC_FILE_NAME = 'prosperLoanData_chargedoff.csv'

#类别型特征
const.CATEGORICAL_FEATURES = ['CreditGrade',  # 信用等级
                              'Term',  #贷款期限，以月为单位
                              'BorrowerState',  #借款人的地址状态，两个字母缩写
                              'Occupation',  #职业
                              'EmploymentStatus',  #就业状态
                              'IsBorrowerHomeowner',  #是否有住房
                              'CurrentlyInGroup',  #是否在某个组织
                              'IncomeVerifiable']  # 是否有收入证明

#数值型特征
const.NUM_FEATURES = ['BorrowerAPR',  # 借款人的贷款年利率
                      'BorrowerRate',  #借款人的贷款利率
                      'LenderYield',  #贷方收益=贷款利率-服务费
                      'ProsperRating (numeric)',  #prosper等级
                      'ProsperScore',  #prosper分数
                      'ListingCategory (numeric)',  #贷款用途
                      'EmploymentStatusDuration',  #雇佣状态持续时间，以月为单位
                      'CurrentCreditLines',  #当前信用等级
                      'OpenCreditLines',  #开放信用等级
                      'TotalCreditLinespast7years',  #过去7年的信用等级
                      'CreditScoreRangeLower',  #信用评分区间中的最低分
                      'OpenRevolvingAccounts',  #循环账户数量
                      'OpenRevolvingMonthlyPayment',  #循环账户月付款
                      'InquiriesLast6Months',  #过去6个月内，信用记录被询问的次数
                      'TotalInquiries',  #信用记录被询问的总次数
                      'CurrentDelinquencies',  #当前的拖欠次数
                      'DelinquenciesLast7Years',  #过去7年的拖欠次数
                      'PublicRecordsLast10Years',  #过去10年的公共记录数量
                      'PublicRecordsLast12Months',  #过去12个月的公共记录数量
                      'BankcardUtilization',  #使用的可循环信贷百分比
                      'TradesNeverDelinquent (percentage)',  #无拖欠交易的百分比
                      'TradesOpenedLast6Months',  #过去6个月的交易次数
                      'DebtToIncomeRatio',  #债务/收入比
                      'LoanFirstDefaultedCycleNumber',  #还贷周期
                      'LoanMonthsSinceOrigination',  #还贷月数
                      'PercentFunded',  #该行用户有效值占比
                      'Recommendations',  #推荐人数量
                      'InvestmentFromFriendsCount',  #做投资的朋友的数量
                      'Investors']  # 为贷款提供资金的投资者的数量

