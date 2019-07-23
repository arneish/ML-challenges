
# coding: utf-8

# In[20]:





# In[37]:


## Import up sound alert dependencies
from IPython.display import Audio, display

def allDone():
  display(Audio(url='https://sound.peal.io/ps/audios/000/000/537/original/woo_vu_luvub_dub_dub.wav', autoplay=True))
## Insert whatever audio file you want above

import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import pandas as pd
import numpy as np
np.random.seed(0)


# In[38]:


data = pd.read_csv("Cristano_Ronaldo_Final_v1/data.csv")
# purged features:
data = data.drop(data.columns[0],axis=1)
data = data.drop(['shot_id_number'],axis=1)

#handling categorical variables (one-hot-encoding)
data = pd.get_dummies(data)
data.head()


# In[49]:


#######LOCAL TRAIN-VALIDATION SPLIT########
#Rescale all feature values:
# mm_scaler = preprocessing.MinMaxScaler()

#global train-test split:
train = data[data['is_goal'].notnull()]
y = train['is_goal']
test = data[data['is_goal'].isnull()]
train = train.fillna(train.mean())
test = test.fillna(test.mean())
train['mean_dist_'] = train.apply(lambda row: math.sqrt(row.location_x*row.location_x+row.location_y*row.location_y), axis=1)
test['mean_dist_'] = test.apply(lambda row: math.sqrt(row.location_x*row.location_x+row.location_y*row.location_y), axis=1)
train.head()


# In[50]:


# local training / validation performance:
train['local_train']=np.random.uniform(0,1,len(train))<=0.85
local_train, local_validation = train[train['local_train']==True], train[train['local_train']==False]
y_local_train = local_train['is_goal']
y_local_validation=local_validation['is_goal']
local_train=local_train.drop(['is_goal'],axis=1)
local_validation =local_validation.drop(['is_goal'],axis=1)

features = local_train.columns[:-1]
local_train = local_train[features]
local_validation = local_validation[features]
# print(local_train.head())

#RESCALING BASED ON STANDARD DISTRIBUTION
scaler = StandardScaler()
scaler.fit(local_train)
local_train = scaler.transform(local_train)
local_validation = scaler.transform(local_validation)
# print("#components_orig:",len(local_train[0]))
# pca = PCA(.90)
# pca.fit(local_train)
# local_train = pca.transform(local_train)
# print("#components:",len(local_train[0]))
# local_validation = pca.transform(local_validation)
print("local train size:", len(local_train))
print("local validation size:", len(local_validation))
print("test size:", len(test))


# In[70]:


#RF Grid Search:
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(20, 70, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True,False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 1, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(local_train, y_local_train)
print(rf_random.best_params_)


# In[42]:


#RF classifier for train-validation perf:
clf = RandomForestClassifier(verbose=1,
                             n_jobs=-1,
                             min_samples_split=2,
                             min_samples_leaf=5,
#                              oob_score=True,
                             bootstrap=False,
                             criterion="entropy",
                             max_depth = 40,
                             n_estimators=1000, random_state=0)
clf.fit(local_train, y_local_train)
p = clf.predict_proba(local_validation)
y_validation_pred_binary = clf.predict(local_validation)
y_validation_pred_prob = []
for x,y in p:
    y_validation_pred_prob.append(y)
count_match = 0
count_error = 0
deviation = 0.0
assert(len(y_validation_pred_prob)==len(y_local_validation))
validation_gtruth=np.asarray(y_local_validation)
for i in range(len(y_local_validation)):
    deviation +=abs(y_validation_pred_prob[i]-validation_gtruth[i])
    if (int(y_validation_pred_binary[i])==int(validation_gtruth[i])):
        count_match+=1
    else:
        count_error+=1
validation_accuracy = count_match/(count_match+count_error)*100.0
print("validation a/c:", validation_accuracy)
print("score:", 1.0/(1.0+deviation*1.0/(count_match+count_error)))
max_depth = list()
for tree in clf.estimators_:
    max_depth.append(tree.tree_.max_depth)
print("avg max depth %0.1f" % (sum(max_depth) / len(max_depth)))
features_imp = pd.DataFrame(clf.feature_importances_, index=features,columns=['importance']).sort_values('importance', ascending=False)
print(features_imp)
allDone()


# In[46]:


#SFM:
sfm = SelectFromModel(clf, threshold=1e-3)
sfm.fit(local_train,y_local_train)
local_train_ = sfm.transform(local_train)
local_validation_ = sfm.transform(local_validation)
clf_ = RandomForestClassifier(n_estimators=1000,random_state=0,criterion="entropy",min_samples_split=2,min_samples_leaf=5,n_jobs=-1, bootstrap=False)
clf_.fit(local_train_, y_local_train)
p = clf_.predict_proba(local_validation_)
y_validation_pred_binary = clf_.predict(local_validation_)
y_validation_pred_prob = []
for x,y in p:
    y_validation_pred_prob.append(y)
count_match = 0
count_error = 0
deviation = 0.0
assert(len(y_validation_pred_prob)==len(y_local_validation))
validation_gtruth=np.asarray(y_local_validation)
for i in range(len(y_local_validation)):
    deviation +=abs(y_validation_pred_prob[i]-validation_gtruth[i])
    if (int(y_validation_pred_binary[i])==int(validation_gtruth[i])):
        count_match+=1
    else:
        count_error+=1
validation_accuracy = count_match/(count_match+count_error)*100.0
print("sfm validation a/c:", validation_accuracy)
print("sfm score:", 1.0/(1.0+deviation*1.0/(count_match+count_error)))
print("num_features:",len(local_train_[0]))
allDone()


# In[51]:


#######ACTUAL TRAIN-TEST PERDICTION########
y_train = train['is_goal']
train=train.drop(['is_goal'],axis=1)
train=train.drop(['local_train'],axis=1)
test =test.drop(['is_goal'],axis=1)

print("Train size:", len(train))
print("Test size:", len(test))

#RF classifier for train-validation perf:
clf2 = RandomForestClassifier(verbose=1,n_estimators=1000,random_state=0,criterion="entropy",min_samples_split=2,min_samples_leaf=5,n_jobs=-1, bootstrap=False)
clf2.fit(train, y_train)
# y_test_pred = clf2.predict(test_modified)
p = clf2.predict_proba(test)
prediction = []
for x,y in p:
    prediction.append(y)


# In[52]:


#write outputs:
count = 0
f = open("submissionAP_2.csv","w+")
print("shot_id_number,is_goal", file=f)
for i in range(len(test_rows)):
    print(str(int(test_rows[i]+1))+","+str(prediction[i]), file=f)
    count+=1
print(count)
f.close()

