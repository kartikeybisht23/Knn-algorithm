import numpy as np
import pandas as pd
from sklearn import datasets
c=datasets.load_iris()
data=c["data"]
y=c["target"]
A=data[0:40]
A_test=data[40:50]
B=data[50:90]
B_test=data[90:100]
C=data[100:140]
C_test=data[140:150]
x_train=np.concatenate((A,B,C))
y_train=np.concatenate((y[0:40],y[50:90],y[100:140]))
len(y_train)
x_test=np.concatenate((data[40:50],data[90:100],data[140:150]))
y_test=np.concatenate((y[40:50],y[90:100],y[140:150]))
#-----------------------------------------
plt.scatter(A[:,0],A[:,1],color="red",label="Class1")
plt.scatter(B[:,0],B[:,1],color="green",label="Class2")
plt.scatter(C[:,0],C[:,1],color="yellow",label="Class3")
plt.legend()
plt.show()
#-------------------------------------------
from itertools import combinations
l=[0,1,2,3]
tup=combinations(l,2)
for x in tup:
    a,b=x
    plt.scatter(A[:,a],A[:,b],color="red",label="Class1")
    plt.scatter(B[:,a],B[:,b],color="green",label="Class2")
    plt.scatter(C[:,a],C[:,b],color="yellow",label="Class3")
    plt.title(f"{a}-{b}")
    plt.legend()
    plt.show()
#--------------------------------------------
fig,ax=plt.subplots(figsize=(12,10),nrows=3,ncols=2)
ax[0][0].scatter(A[:,0],A[:,1],color="red",label="Class1")
ax[0,0].scatter(B[:,0],B[:,1],color="green",label="Class2")
ax[0][0].scatter(C[:,0],C[:,1],color="yellow",label="Class3")
ax[0,0].legend()
#---
ax[0][1].scatter(A[:,0],A[:,2],color="red",label="Class1")
ax[0,1].scatter(B[:,0],B[:,2],color="green",label="Class2")
ax[0][1].scatter(C[:,0],C[:,2],color="yellow",label="Class3")
ax[0,1].legend()
#---
ax[1][0].scatter(A[:,0],A[:,3],color="red",label="Class1")
ax[1,0].scatter(B[:,0],B[:,3],color="green",label="Class2")
ax[1][0].scatter(C[:,0],C[:,3],color="yellow",label="Class3")
ax[1,0].legend()
#---
ax[1][1].scatter(A[:,1],A[:,2],color="red",label="Class1")
ax[1,1].scatter(B[:,1],B[:,2],color="green",label="Class2")
ax[1][1].scatter(C[:,1],C[:,2],color="yellow",label="Class3")
ax[1,1].legend()
#---
ax[2][0].scatter(A[:,1],A[:,3],color="red",label="Class1")
ax[2,0].scatter(B[:,1],B[:,3],color="green",label="Class2")
ax[2][0].scatter(C[:,1],C[:,3],color="yellow",label="Class3")
ax[2,0].legend()
#---
ax[2][1].scatter(A[:,2],A[:,3],color="red",label="Class1")
ax[2,1].scatter(B[:,2],B[:,3],color="green",label="Class2")
ax[2][1].scatter(C[:,2],C[:,3],color="yellow",label="Class3")
ax[2,1].legend()
#-----------------------------------------------------
#part 6
def cal(pt1,pt2):
    return(np.sum((pt1-pt2)**2))**0.5

def classify(train_data,test_pt):
    total=[]
    for i in train_data:
        sumt=cal(i,test_pt)
        total.append(sumt)
    return total
a=classify(x_train,x_test[0])
def nearest_point(x_train,y_train,test_pt,k=30):
    distance=classify(x_train,test_pt)
    idx=np.argsort(distance)
    nn=idx[:30]
    return y_train[nn]

a=nearest_point(x_train,y_train,x_train[0]) 
#part7
def accuracy(x_train,y_train,x_test,y_test):
    pred=[]
    for i in x_test:
        nearest=nearest_point(x_train,y_train,i)
        unique,count=np.unique(nearest,return_counts=True)
        idx_ans=np.argsort(count)
        pred.append(unique[idx_ans[-1]])
    pred_arr=np.array(pred)
    answer=np.where(y_test==pred_arr,1,0)
    print(y_test,pred_arr)
    return((np.sum(answer)/30)*100)

accuracy(x_train,y_train,x_test,y_test)  
