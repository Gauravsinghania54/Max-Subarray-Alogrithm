#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from kmodes.kmodes import KModes
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# In[2]:


# random categorical data
df=pd.read_csv('cars.csv', delimiter=',', usecols=['Type', 'Origin','DriveTrain','Cylinders'])
df


# In[3]:


df['Cylinders'].fillna(0.0, inplace=True)


# In[4]:


df['Type'].value_counts()


# In[5]:


df['DriveTrain'].value_counts()


# In[6]:


origin=df['Origin'].value_counts()
origin


# In[7]:


distance_Origin=((1/origin['Asia'])+(1/origin['Europe']))
distance_Origin


# In[8]:


cylinder=df['Cylinders'].value_counts()
cylinder


# In[9]:


distance_Cylinders=((1/cylinder.loc[5.0])+(1/cylinder.loc[0.0]))
distance_Cylinders


# # DataFrame to Numpy Array

# In[10]:


arr=df.to_numpy()

arr=np.where(arr=='SUV', 60, arr)
arr=np.where(arr=='Sedan', 262, arr)
arr=np.where(arr=='Sports', 49, arr)
arr=np.where(arr=='Wagon', 30, arr)
arr=np.where(arr=='Truck', 24, arr)
arr=np.where(arr=='Hybrid', 3, arr)
arr=np.where(arr=='FWD', 226, arr)
arr=np.where(arr=='AWD', 92, arr)
arr=np.where(arr=='RWD', 110, arr)
arr=np.where(arr=='Asia', 158, arr)
arr=np.where(arr=='USA', 147, arr)
arr=np.where(arr=='Europe', 123, arr)
arr=np.where(arr==6.0, 190, arr)
arr=np.where(arr==4.0, 136, arr)
arr=np.where(arr==8.0, 87, arr)
arr=np.where(arr==5.0, 7, arr)
arr=np.where(arr==12.0, 3, arr)
arr=np.where(arr==0.0, 2, arr)
arr=np.where(arr==10.0, 2, arr)
arr=np.where(arr==3.0, 1, arr)
arr


# # Kmodes Algorithm with distance and mode methods

# In[19]:


def dist_between(val,k):
    if(len(val)==0 or len(k)==0):
         return 0;
    d=0;
    if(val.all()==k.all()):
        return 0;
    for i in range (len(k)):
        d=d+((1/k[i])+(1/val[i]))
    return d
def mode_calculate(a):
    x=stats.mode(a)
    y=np.asarray(x)
    r=y[0]
    return r.transpose()
    

k1=arr[10]
k2=arr[50]
k3=arr[200]
k1_clus=[]
k2_clus=[]
k3_clus=[]
for i in range(10):
    k1_clus=[]
    k2_clus=[]
    k3_clus=[]
    
    for j in range(arr.shape[0]):
        m=arr[j]
       
        t1=dist_between(m,k1)
        t2=dist_between(m,k2)
        t3=dist_between(m,k3)
        t4=min(t1,t2,t3)
        if t4==t1:
            #m is assigned to k1 clusture
            #print("me")
            k1_clus.append(m)
        elif t4==t2:
            #m is assigned to k2 clusture
            k2_clus.append(m)
        else :
            #m is assigned to k3 clustre
            k3_clus.append(m)
        #print(k1_clus)
    
    k1_new=mode_calculate(np.asarray(k1_clus))
    k2_new=mode_calculate(np.asarray(k2_clus))
    k3_new=mode_calculate(np.asarray(k3_clus))
    
    #print(np.asarray(k1_clus))
    #print(np.asarray(k2_clus))
    #print(np.asarray(k3_clus))
    #print(dist_between(k1,k1_new))
    #print(dist_between(k2,k2_new))
    #print(dist_between(k3,k3_new))
    #print("\n")          
    k1=k1_new
    k2=k2_new
    k3=k3_new
    
print("the length of first cluster",len(k1_clus))
print("the length of second clusterlen",len(k2_clus))
print("the length of Third clusterlen",len(k3_clus))

print("The Centroid of the cluster k1\n",k1)
print("The Centroid of Cluster K2\n",k2)
print("The Centroid of Cluster K3\n",k3)


# # cluster 1 origin distribution

# In[13]:


cl=[]
cl_asia=0
cl_europe=0
cl_usa=0
for i in range(len(k1_clus)):
    cl.append(k1_clus[i][1])
    if k1_clus[i][1]==158:
        cl_asia+=1
    elif k1_clus[i][1]==123:
        cl_europe+=1
    else:
        cl_usa+=1

print(f"Frequency Distribution  of Cluster 1: Asia :{cl_asia}, Europe:{cl_europe}, USA:{cl_usa}")

#Cluster 1 Frequency Distribuation  
  
plt.hist(cl,edgecolor='k')
plt.ylabel("Number of Occurance of Given Frequency")
plt.xlabel("Frequency of Origin")
plt.title("Cluster 1")
plt.show()
plt.clf() 


# # cluster 2 Origin Distribution

# In[15]:


c2=[]
c2_asia=0
c2_europe=0
c2_usa=0
for i in range(len(k2_clus)):
    c2.append(k2_clus[i][1])
    if k2_clus[i][1]==158:
        c2_asia+=1
    elif k2_clus[i][1]==123:
        c2_europe+=1
    else:
        c2_usa+=1

print(f"Frequency Distribution  of Cluster 2: Asia :{c2_asia}, Europe:{c2_europe}, USA:{c2_usa}")

#Cluster 2 Frequency Distribuation  
  
plt.hist(c2,edgecolor='k')
plt.ylabel("Number of Occurance of Given Frequency")
plt.xlabel("Frequency of Origin")
plt.title("Cluster 1")
plt.show()
plt.clf() 


# In[17]:


c3=[]
c3_asia=0
c3_europe=0
c3_usa=0
for i in range(len(k3_clus)):
    c3.append(k3_clus[i][1])
    if k3_clus[i][1]==158:
        c3_asia+=1
    elif k3_clus[i][1]==123:
        c3_europe+=1
    else:
        c3_usa+=1

print(f"Frequency Distribution  of Cluster 3: Asia :{c3_asia}, Europe:{c3_europe}, USA:{c3_usa}")
#Cluster 3 Frequency Distribuation  
  
plt.hist(c3,edgecolor='k')
plt.ylabel("Number of Occurance of Given Frequency")
plt.xlabel("Frequency of Origin")
plt.title("Cluster 1")
plt.show()
plt.clf() 


# In[ ]:




