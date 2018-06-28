import  pandas as pd
import  pickle

from  settings import  *

with open(ROOT_DIR + 'featureEngineering/numericalFeatures.pkl', 'rb') as f1:
    numericalFeatures = pickle.load(f1)
    print(numericalFeatures)
with open(ROOT_DIR + 'featureEngineering/bin_dict.pkl', 'rb') as f2:
    bin_dict = pickle.load(f2)
    print(bin_dict)
with open(ROOT_DIR + 'featureEngineering/categoricalFeatures.pkl', 'rb') as f3:
    categoricalFeatures = pickle.load(f3)
    #print(categoricalFeatures)

with open(ROOT_DIR + 'featureEngineering/featuresInModel.pkl', 'rb') as f4:
    featuresInModel = pickle.load(f4)
    #print('features In Model: ')
    #print(featuresInModel)

with open(ROOT_DIR + 'featureEngineering/WOE_IV_dict.pkl', 'rb') as f5:
    WOE_IV_dict = pickle.load(f5)

print('WOE_IV_dict:')
print(WOE_IV_dict)

modelFeatures = [i.replace('_Bin','').replace('_WOE','') for i in featuresInModel]

# print('model features:')
# print(modelFeatures)
#
# numFeatures = [i for i in modelFeatures if i in numericalFeatures]
# charFeatures = [i for i in modelFeatures if i in categoricalFeatures]
#
# print('numfeatures:')
# print(numFeatures)
#
# print('charfeatures:')
# print(charFeatures)
#
# finalFeatures = [i+'_Bin' for i in numFeatures] + ['M2FreqL3M_Bin','maxDelqL1M_Bin','maxDelqL3M_Bin','M0FreqL3M']

for var in numericalFeatures:
    newBin = var + "_Bin"
    #bin = [i.values() for i in bin_dict if var in i][0][0]
    # for i in bin_dict:
    #     #print(i)
    #     if var in i:
    #         print(var + '   is in:', i)
    #         print(i[var])
    #print(bin)
    print(len(bin_dict))
    bin = [i[var] for i in bin_dict if var in i][0]
    print(bin)

