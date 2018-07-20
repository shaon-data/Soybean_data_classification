# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "D:\work\codes\Ripositories\Data Science\My_Lib\EDA")
import numpy as np
import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt
import EDA as ed
from matplotlib import style
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
style.use('ggplot')
sns.set()


names=['class', 'date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged', 'severity', 'seed-tmt', 'germination', 'plant_growth', 'leaves', 'leafspots_halo', 'leafspots_marg', 'leafspot_size', ' leaf_shread', 'leaf_malf', 'leaf_mild', 'stem', 'lodging', 'stem_cankers', 'canker_lesion', 'fruiting_bodies', 'external_decay', 'mycelium', 'int_discolor', 'sclerotia', 'fruit_pods', 'fruit_spots', 'seed', 'mold_growth', 'seed_discolor', 'seed_size', 'shriveling', 'roots']
## Loading Data
dat = pd.read_csv('data/soybean-large.csv',names=names)

data = dat.copy()
data = data.drop(['class'],1)




## Replacing missing value '?' with -1
data.replace('?',0,inplace=True)


'''
## String to Integer coversion of class label
class_label_str = data['class'].unique().tolist()
#### No label missing so started from 0 by range
class_label_int = [c for c in range(len(class_label_str))]
for c in class_label_str:
    data[data['class'] == c] = class_label_int[ class_label_str.index(c) ]
'''
## Converting all column to integer datatype
data = data.astype('int')




## Optimization: experimenting with differnt K values with their model costs
k_s = []
costs = []
nLabels = []

X = data
for k in range(1,60): ## experiment with n
    if True: ## Dont use Odd logic - if it is not continuous, we will not able to produce the real result
        ## Initializing model with a fixed random seed
        clusters = KMeans(n_clusters=k, random_state = 1)
        clusters.fit(X)

        ## Getting predicted Labels
        predictedLabelY = clusters.labels_

        ## Getting Model cost/inertia/sum of squared distance of data points from centroid
        cost = clusters.inertia_

        ## Genarating col name of K value for predicted labels
        col_name = 'k'+str(k)+'_label'
        ## Saving predicting labels

        data[col_name] = predictedLabelY
        ## Number of labels for specific K value

        ## Saving k value in every session
        k_s.append(k)
        ## Saving Number of labels for specific K value
        nLabels.append(data[col_name].nunique())
        ## Saving Cost or inertia for specific K value of clustering model
        costs.append(cost)

k_ticks = ["k"+str(k) for k in k_s]
#ind = np.arange(len(range(2,15)))



## All possibilities with value of K

## shifting indexes to 1 row down
## data.index += 1

## Saving the labeled Result
data.to_csv('result/unsupervised_label.csv')

## Plotting the k vs Number of labels to understand the cluster
plt.figure("k vs Number of labels")
plt.plot(k_s,nLabels, marker = 'x')
plt.title("k vs label numbers")
plt.xlabel('K')
plt.ylabel('Number of labels')
plt.savefig("result/k_vs_Number_of_labels.png")

    
## Plot of  Optimization starts
plt.figure("k vs Model Cost and k vs Change rate in Model Cost")
## Plotting the k vs Model cost
#plt.figure("k vs Model Cost(sum of distance from centroid)")
plt.subplot(3,1,1)
plt.plot(k_s,costs, marker = 'x')
plt.title("Title:k vs Model Cost(sum of distance from centroid)")
plt.xlabel('k')
plt.ylabel('Model Cost')


##d/dk(costs) = slope of Costs reference to K value = Rate of change of Costs reference to change of x
## M = slope_list_curve(k_s,costs)
from numpy import diff
print(len(costs),len(k_s))
M = diff(costs)/diff(k_s)
k_s=k_s[1:]
M1 = np.absolute(M - np.median(M))    
    
## Visualizing optimized K value
plt.subplot(3,1,2)
#plt.figure("k vs d/dk(Cost)")
plt.plot(k_s,M, marker = 'x')
plt.title("Title:k vs Change_rate(Cost)")
plt.xlabel('k')
plt.ylabel('Change in Cost(2)')
    

M = diff(M)/diff(k_s)
k_s=k_s[1:]
M2 = np.absolute(M - np.median(M))

## Visualizing optimized K value
plt.subplot(3,1,3)
#plt.figure("k vs d/dk(Cost)")
plt.plot(k_s,M, marker = 'x')
plt.title("Title:k vs Change_rate(Cost)2")
plt.xlabel('k')
plt.ylabel('Change in Cost')
    
plt.tight_layout()
plt.savefig("result/kcost_ddk_costs.png")
plt.show()
## Plot of  Optimization ends



M= M.tolist()
best_k_index = M.index(min(M))
best_k = k_s[best_k_index]
best_cluster_number = nLabels[best_k_index]

print(best_cluster_number)
M1 = M1.tolist()
M2 = M2.tolist()
print( nLabels[M2.index(min(M2))] - nLabels[M1.index(min(M1))])



'''
clf = KMeans(n_clusters=best_cluster_number)
clf.fit(X)
## For Kaggle
print(clf.score(X,data.ix[:,0]))
'''
    
    



