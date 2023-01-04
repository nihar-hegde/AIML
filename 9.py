import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def local_regression(x0,X,Y,tau):
    x0=np.r_[1,x0]
    X=np.c_[np.ones(len(X)),X]
    xw=X.T *radial_kernel(x0,X,tau)
    print(xw)
    beta=np.linalg.pinv(xw @ X)@ xw @ Y
    return x0@beta
print(np.r_[np.array([1,2,3]),0,0,0,np.array([4,5,6])])
print(np.c_[np.array([1,2,3]),np.array([4,5,6])])
def radial_kernel(x0, X, tau):
    return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau * tau))
data=pd.read_csv("data10_tips.csv")
bill=data.total_bill.values
tip=data.tip.values
tau=10
ypred=np.array([local_regression(x0,bill,tip,tau) for x0 in bill])
SortIndex=bill.argsort(0)
xsort=bill[SortIndex]
plt.figure(figsize=(10,5))
plt.scatter(bill,tip)
plt.plot(xsort,ypred[SortIndex],color='red',linewidth=5)
plt.scatter(bill,tip,color='green')
plt.xlabel('Total Bill')
plt.ylabel("Tip")
plt.show()
