import  numpy as np

vc=[1,2,39,0,8]
vb=[1,2,100,0,125]


result = np.corrcoef(vc,vb)
print(result)
print(result[1, 0])