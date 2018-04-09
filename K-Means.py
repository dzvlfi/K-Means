
# coding: utf-8

# In[1]:


#Dzulfiqar Ridha 1301154298 IF-39-04
import matplotlib.pyplot as plt
import numpy as np
import math
import random


# In[2]:


#load data
data_train = np.genfromtxt("TrainsetTugas2.txt")
data_test = np.genfromtxt("TestsetTugas2.txt")

#visualization train data
x = data_train[:,0]
y = data_train[:,1]

plt.scatter(x, y, alpha=0.5)
plt.show()


# In[3]:


#fungsi visualisasi data
def VisualResult(data):
    x = data[:,0]
    y = data[:,1]
    colors = data[:,2]

    plt.scatter(x, y, c=colors)
    return plt


# In[4]:


#hitung Euclidean Distance
def Euclid(x1,y1,x2,y2):
    return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))


# In[5]:


#fungsi untuk generate centroid dg koordinat x,y dari data train
def TrainMakeCentro(k):
    centro = {} #cluster
    for i in range(k):
        centro[i] = np.array(random.choice(data_train))
        
    for i in range(100):
        old, new = NewCentroidandLabels(1,data_train,centro)
        
    centro[0] = np.array([6.2,4.25])
    centro[0] = np.array([34.5,5.05])
    centro[0] = np.array([7.35,24.75])
    centro[0] = np.array([17.4,6.5])
    centro[0] = np.array([21.9,23.65])    
    
    return new


# In[6]:


#mencari label berdasarkan jarak terdekat
def FindLabel(data, centroid):
    label = []
    for i in data:
        tmp = 50
        for j in range(5):
            eucl = Euclid(i[0],i[1],centroid[j][0],centroid[j][1])
            if tmp > eucl:
                tmp = eucl
                id = j
        label.append(id)
    return np.array(label)


# In[7]:


#Mencari Centroid yang baru
def NewCentroidandLabels(train, data, cent):
    label = FindLabel(data, cent)
    data_label = np.concatenate((data,label[:,None]),axis=1)

    loc = {}
    for i in range(5):
        loc[i] = []
        for j in data_label:
            if (i == j[2]):
                loc[i].append(j[0:2])

    centNew = {}
    for i in range(5):
        centNew[i] = np.array([np.mean(loc[i][0]),np.mean(loc[i][1])])
    
    if (train == 1):
        return cent, centNew
    else:
        return cent, centNew, data_label


# In[8]:


#klasterisasi data berdasarkan centroid yang sudah ada
def Klaster(data,centroid):
    label = FindLabel(data, centroid)
    data_label = np.concatenate((data,label[:,None]),axis=1)
        
    return data_label


# In[9]:


#main program
if __name__ == '__main__':
    centroid = TrainMakeCentro(5)
    result = Klaster(data_test,centroid)
    plot = VisualResult(result)
    plot.show()
    np.savetxt('result.txt', (result), fmt="%d")

