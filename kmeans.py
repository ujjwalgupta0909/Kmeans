# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:04:06 2019

@author: hp
"""

#KMEANS

#IMPORTING THE LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt

#getting the dataset
dataset=pd.read_csv('Mall_customers.csv') 
features=dataset.iloc[:,[3,4]].values

#applying the elbow method
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
#SO NOW ACC. TO ELBOW METHOND WE CHOOSE THE NUMBER OF CLUSTERS TO BE 5

#APPLY K MEANS
kmeans=KMeans(n_clusters=5,random_state=0,init='k-means++')
pred=kmeans.fit_predict(features)

#VISUALIZING
plt.scatter(features[pred==0,0],features[pred==0,1],color='red',s=100,label='cluster1')
plt.scatter(features[pred==1,0],features[pred==1,1],color='green',s=100,label='cluster2')
plt.scatter(features[pred==2,0],features[pred==2,1],color='blue',s=100,label='cluster3')
plt.scatter(features[pred==3,0],features[pred==3,1],color='orange',s=100,label='cluster4')
plt.scatter(features[pred==4,0],features[pred==4,1],color='black',s=100,label='cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,label='centroid',color='yellow')
plt.legend()
plt.title('kmeans')
plt.xlabel('income')
plt.ylabel('spending score')
plt.show

    