
# coding: utf-8

# ## Importing the librariest and settings

# In[10]:


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


# ## Loading the data

# In[11]:


names=['class', 'date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged', 'severity', 'seed-tmt', 'germination', 'plant_growth', 'leaves', 'leafspots_halo', 'leafspots_marg', 'leafspot_size', ' leaf_shread', 'leaf_malf', 'leaf_mild', 'stem', 'lodging', 'stem_cankers', 'canker_lesion', 'fruiting_bodies', 'external_decay', 'mycelium', 'int_discolor', 'sclerotia', 'fruit_pods', 'fruit_spots', 'seed', 'mold_growth', 'seed_discolor', 'seed_size', 'shriveling', 'roots']
## Loading Data
dat = pd.read_csv('data/soybean-large.csv',names=names)

data = dat.copy()
data = data.drop(['class'],1)
print(data)


# ## Formating missing data and coverting to a integer dataframe

# In[12]:


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
print(data)


# ## Data Base Shape and column DTypes

# In[13]:


print("|-------- Dataset information --------|")
shape = data.shape
print("Shape "+str(shape))
print("Data type: \n",data.dtypes)


# ## String charatecter check

# In[14]:


def string_column_count(x):
    return len(x) - sum([ str(c).lstrip("-").isdigit() for c in x])
        
print("String column count:\n", data.apply( lambda x: string_column_count(x) ,axis = 0))


# ## Checking Corelations

# In[15]:


ed.correlation_sorted(data)


# ## Correlation Matrix hit map

# In[16]:


correlations = data.corr()
# plot correlation matrix
fig = plt.figure('Correlation Hit map')
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()

## ed.scatter_matrix_graph_fit(data)


# ## Checking the columns with normal distribution

# In[17]:


## Fill not available value from the skewness probability distribution and mode ,median, mean and skewness and kurtosis and chi square test
## coefficent_of_skewness(data)

mode,mode_count = stat.mode(data,axis=0)
print("Mode: "+ str( mode[0] ) + "\n")
print("Mean: \n" + str( np.mean(data,axis=0) ) + "\n" )
print("Median: "+ str( np.median(data,axis=0) ) + "\n" )

print("For normally distributed data, the skewness should be about 0. For unimodal continuous distributions, a skewness value > 0 means that there is more weight in the right tail of the distribution. The function skewtest can be used to determine if the skewness value is close enough to 0, statistically speaking.")
print("Coefficient of skewness : \n" + str( stat.skew(data, axis=0, bias=False) ) + "\n")



print("Moment coefficient kurtosis = 3 , meso kurtic & normal distribution\n> 3 , lepto kurtic\n< 3 , platy kurtic")
print("Coefficient of kurtosis : \n" + str( stat.kurtosis(data,axis=0,fisher=False,bias=False) ) + "\n")
## If False, then the calculations are corrected for statistical bias.

# ?? Pearson Chi square test for data comparing to statistical distribution fit


# In[18]:



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
data.to_csv('unsupervised_label.csv')

## Plotting the k vs Number of labels to understand the cluster
plt.figure("k vs Number of labels")
plt.plot(k_s,nLabels, marker = 'x')
plt.title("k vs label numbers")
plt.xlabel('K')
plt.ylabel('Number of labels')
plt.savefig("k_vs_Number_of_labels.png")

    
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
plt.savefig("kcost_ddk_costs.png")
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


# ![title](EXCEL_FORECAST.png)

# # As We can See,
# ### In the first graph, Number of labels always increases as the number of K increases. And they are always equal
# ### In the second graph, we ploted K vs Model Cost, K vs ddK(Model Cost) and K vs ddK(Model Cost)^2 .
# ### As we know there is no direct way to pick the best value for K, so, we have to pick it visually.
# ### We can see when the value K = 20 , almost all of the 3 graph's value in Y axis almost stop changing.
# ### And we have estimated we should have natural k value of sqrt(sample_number) = sqrt(307) = about 18
# ### We can now estimated that we should have cluster between 18 - 20. Which should be 19 
# ### For confirming , we have forcasted our number of labels in MS excel, by using labels of K2 - K29. Whis is 19
