#group:daniahilal 201911486
#      rinat imad 201911443
#data:Loan Prediction Based on Customer Behavior
print("------------------------------------------------------------------------------------")
print("group:daniahilal 201911486 ")
print("      rinat imad 201911443")
print("data:Loan Prediction Based on Customer Behavior")
print("------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------")


import imp
import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

#loading my dataset
data=pd.read_csv(r"C:\Users\L\Desktop\sem3\data mining\مجلد جديد\Training Data.csv")

# missing handling by deletting
print("data before handling missing index",len(data))
data = data.dropna(axis=0,how='any')
print("data after handling missing index",len(data))
print("------------------------------------------------------------------------------------")


#remove duplicated
dups = data.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))
print("data before removing duplicated index",len(data))
data=data.drop_duplicates()
print("data after removing duplicated index",len(data))
print("------------------------------------------------------------------------------------")



#Remove irrelevant attributes depend on relation with risk flag attribute
#first i Convert Categorical Variables to Numerical
#get all categorical columns
#numerical dataset
nudata=data

cat_columns = data.select_dtypes(['object']).columns
#convert all categorical columns to numeric
nudata[cat_columns] = nudata[cat_columns].apply(lambda x: pd.factorize(x)[0])
cor = nudata.corr()
cor_target = abs(cor["Risk_Flag"])
#detect features with less depend on risk flag"my class"
irrelevant_features = cor_target [cor_target < 0.01]
print ("#of irrr features",len(irrelevant_features))
print("irrr features",irrelevant_features.index)
#delete irrlevent
nudata=data.drop(labels=irrelevant_features.index, axis=1)
data=data.drop(labels=irrelevant_features.index, axis=1)
print ("columns after drop irrelevent ones",data.columns)
print("------------------------------------------------------------------------------------")


#Remove correlated attributes.
cat_columns = data.select_dtypes(['object']).columns
#convert all categorical columns to numeric
nudata[cat_columns] = nudata[cat_columns].apply(lambda x: pd.factorize(x)[0])
print("befor drop corr",nudata.shape,nudata.columns)

correlated_features = set()
correlation_matrix = nudata.corr()
print(nudata.corr())
for i in range(len(correlation_matrix .columns)):
    for j in range(i) :
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
print("#of dependent featurs ",len(correlated_features))
nudata.drop(columns=correlated_features, axis=1, inplace=True)
data.drop(columns=correlated_features, axis=1, inplace=True)
print("data after corrolation drop",data.columns)
print("------------------------------------------------------------------------------------")



#remove noise using z-score
from scipy import stats
z_scores = stats.zscore(data)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
new_df = data[filtered_entries]
print('data after remove noise')
print(new_df)
print("------------------------------------------------------------------------------------")




#discretization on numeric attributes 
print (data)
num_colums = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = list(data.select_dtypes(include=num_colums).columns)

for i in range(len(data[numerical_columns].columns)-1):
     bins = pd.qcut(data[numerical_columns[i+1]],2000,duplicates = 'drop')
     bins.value_counts(sort=False)
     print("data after discretization",bins.value_counts(sort=False))
print("------------------------------------------------------------------------------------")



#split the data into train 80% and test set20%
train,test = train_test_split(data, test_size=0.20, random_state=0)
#save the data ino separate files 
train.to_csv('training.csv',index=False)
test.to_csv('testing.csv',index=False)
print("finished first part")
print("------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------")

#classification
# KNN Model classification algorithm 1
accuracies = {}
import matplotlib.pyplot as plt

data[cat_columns] = data[cat_columns].apply(lambda x: pd.factorize(x)[0])

y = data.Risk_Flag
x= data.drop(['Risk_Flag'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
#transpose matrices
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T
# KNN Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(x_train.T, y_train.T)
prediction = knn.predict(x_test.T)

print("{} NN Score: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))
# try ro find best k value
scoreList = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(x_train.T, y_train.T)
    scoreList.append(knn2.score(x_test.T, y_test.T))
acc = max(scoreList)*100
accuracies['KNN'] = acc
print("Maximum KNN Score is {:.2f}%".format(acc))

print("--------------------")


#Naive Bayes classsification algorithm 2
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train.T, y_train.T)
acc = nb.score(x_test.T,y_test.T)*100
accuracies['Naive Bayes'] = acc
print("Accuracy of Naive Bayes: {:.2f}%".format(acc))

print("--------------------")


#comparing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
dtc = DecisionTreeClassifier()
knc = KNeighborsClassifier()

keys = ['DecisionTreeClassifier','KNeighborsClassifier']
values = [DecisionTreeClassifier(), KNeighborsClassifier()]
scores = []
for value in values:
    model=value
    model.fit(x_train,y_train)
    predict = model.predict(x_test)
    acc = accuracy_score(y_test, predict)
    scores.append(acc)
plt.figure(figsize = (8,2))
sns.barplot(x = scores, y = keys, palette='muted')
plt.title("Model Scores", fontsize=16, fontweight="bold")


print("------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------")

#clustering
# k-means clustering algorithm 1
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=10000, n_features=6, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = KMeans(n_clusters=2)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()

print("----------------------------")

#Single Link (MIN) clustering algorithm 2
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
Z = hierarchy.linkage(x_train.to_numpy(), 'single')
dn = hierarchy.dendrogram(Z,orientation='right')


print("------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------")



# association rules
# get association rules by using FP-growth algorithm 1
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder


te = TransactionEncoder()

#Transform the transection dataset to binary 20 array

te_ary= te.fit(data).transform(data)

print(te_ary)
#covert the array of transaction data array into pandas DataFrame
df = pd.DataFrame(te_ary)

#get the frequent itemsets by using apriori algorithm
frequentItemsets = apriori(df, min_support=0.6, use_colnames=True)
print('Itemsets\n', frequentItemsets)

# get the association rules-
rules= association_rules (frequentItemsets, min_threshold=0.7) 
print('Rules\n', rules)


print("----------------------------")

# get association rules algorithm 2
import pyfpgrowth
#use FP-growth to get patterns with minimum support = 3
patterns= pyfpgrowth.find_frequent_patterns(data,3)

#use FP-growth to get association rules with minimum confidence = 0.7

rules = pyfpgrowth.generate_association_rules(patterns, 0.7) 
print("Rules\n", rules)
 
print("------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------")

