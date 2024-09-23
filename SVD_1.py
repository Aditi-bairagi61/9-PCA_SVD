# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:18:46 2024

@author: HP
"""
import numpy as np
from numpy import array
from scipy.linalg import svd
A=array([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,4,0,0,0]])
print(A)
#SVD
U,d,Vt=svd(A)
print(U)
print(d)
print(Vt)
print(np.diag(d))
#SVD applying to the dataset
import pandas as pd
data=pd.read_excel("D:/DS/9-PCA,SVD/University_Clustering.xlsx")
data.head()
data=data.iloc[:,2:] #removes non numeric data
data 

from sklearn.decomposition import TruncatedSVD
svd=TruncatedSVD(n_components=3)
svd.fit(data)
result=pd.DataFrame(svd.transform(data))
result.head()
result.columns="pc0","pc1","pc2"
result.head()
#scatter diagram
import matplotlib.pyplot as plt
plt.scatter(x=result.pc0,y=result.pc1)
#pc0 should have higher value compare to pc1
#graph shoould be \ like this 
