import numpy as np
import re
import time
import datetime

from dateutil.relativedelta import relativedelta


def CareerYear(x):
    #对工作年限进行转换
    if x.find('n/a') > -1:
        return -1
    elif x.find("10+")>-1:   #将"10＋years"转换成 11
        return 11
    elif x.find('< 1') > -1:  #将"< 1 year"转换成 0
        return 0
    else:
        return int(re.sub("\D", "", x))   #其余数据，去掉"years"并转换成整数


def DescExisting(x):
    #将desc变量转换成有记录和无记录两种
    if type(x).__name__ == 'float':
        return 'no desc'
    else:
        return 'desc'


def ConvertDateStr(x):
    mth_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
                'Nov': 11, 'Dec': 12}
    if str(x) == 'nan':
        return datetime.datetime.fromtimestamp(time.mktime(time.strptime('9900-1','%Y-%m')))
        #time.mktime 不能读取1970年之前的日期
    else:
        yr = int(x[4:6])
        if yr <=17:    #默认小于17年的是21世纪
            yr = 2000+yr
        else:
            yr = 1900 + yr  #默认大于17年的是1900多年的
        mth = mth_dict[x[:3]]
        return datetime.datetime(yr,mth,1)    #1970年以前的直接使用datetime


def MonthGap(earlyDate, lateDate):
    if lateDate > earlyDate:
        gap = relativedelta(lateDate,earlyDate)  #计算时间跨度
        yr = gap.years
        mth = gap.months
        return yr*12+mth
    else:
        return 0


def MakeupMissing(x):
    if np.isnan(x):
        return -1
    else:
        return x


def ModifyDf(x, new_value):
    if np.isnan(x):
        return new_value
    else:
        return x
