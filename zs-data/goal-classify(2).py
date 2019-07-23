
# coding: utf-8

# In[33]:


## Import up sound alert dependencies
# from IPython.display import Audio, display

# def allDone():
#   display(Audio(url='https://sound.peal.io/ps/audios/000/000/537/original/woo_vu_luvub_dub_dub.wav', autoplay=True))
# ## Insert whatever audio file you want above

import math
from sklearn.ensemble import ExtraTreesRegressor
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


# In[34]:


data = pd.read_csv("Cristano_Ronaldo_Final_v1/data.csv")
# purged features:
data = data.drop(data.columns[0],axis=1)
# data = data.groupby(data.columns, axis=1)
data = data.drop(['shot_id_number'],axis=1)
data = data.drop(['date_of_game'],axis=1)
data = data.drop(['team_id'],axis=1)

#handling categorical variables (one-hot-encoding)
data = pd.get_dummies(data)
# data = data.groupby(data.columns, axis=1).mean()
data.head()


# In[35]:


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
# train = train.groupby(train.columns, axis=1).agg(np.mean)
# test = test.groupby(test.columns, axis=1).agg(np.mean)
train.head()


# In[36]:


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
# scaler = StandardScaler()
# scaler.fit(local_train)
# local_train = scaler.transform(local_train)
# local_validation = scaler.transform(local_validation)
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
# from sklearn.model_selection import RandomizedSearchCV
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(20, 70, num = 10)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True,False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# print(random_grid)

# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestClassifier()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 1, verbose=2, random_state=42, n_jobs = -1)
# # Fit the random search model
# rf_random.fit(local_train, y_local_train)
# print(rf_random.best_params_)


# In[ ]:


#RF classifier for train-validation perf:
clf = ExtraTreesRegressor(verbose=2, n_jobs=1,oob_score=True,min_samples_leaf=2, bootstrap=True,criterion='mae', max_depth = 30, n_estimators=200, random_state=0)
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
# allDone()


# In[ ]:


#SFM:
sfm = SelectFromModel(clf, threshold=1e-3)
sfm.fit(local_train,y_local_train)
local_train_ = sfm.transform(local_train)
local_validation_ = sfm.transform(local_validation)
# clf_ = RandomForestRegressor(n_estimators=500,random_state=0, n_jobs=7)
clf_ = ExtraTreesRegressor(n_jobs=1,oob_score=True,min_samples_leaf=2, bootstrap=True,criterion='mae', max_depth = 30, n_estimators=200, random_state=0)
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
# allDone()
exit(0)

# In[11]:





# In[ ]:


#######ACTUAL TRAIN-TEST PERDICTION########
y_train = train['is_goal']
train=train.drop(['is_goal'],axis=1)
test =test.drop(['is_goal'],axis=1)
features = local_train.columns[:-1]
train = train[features]
test = test[features]
print("Train size:", len(train))
print("Test size:", len(test))

# preprocessing local-train+validation:
# imp= SimpleImputer(missing_values=np.nan,strategy='mean')
# imp = imp.fit(train)
# train_modified = imp.transform(train)
# imp = imp.fit(test)
# test_modified = imp.transform(test)

#RF classifier for train-validation perf:
clf2 = RandomForestClassifier(n_jobs=2, n_estimators=100, random_state=0)
clf2.fit(train_modified, y_train)
# y_test_pred = clf2.predict(test_modified)
p = clf2.predict_proba(test_modified)
prediction = []
for x,y in p:
    prediction.append(y)
count_match = 0
count_error = 0

# assert(len(y_test_pred)==len(y_local_validation))
# validation_gtruth=np.asarray(y_local_validation)
# for i in range(len(y_local_validation)):
#     if (int(y_validation_pred[i])==int(validation_gtruth[i])):
#         count_match+=1
#     else:
#         count_error+=1
# validation_accuracy = count_match/(count_match+count_error)*100.0
# print("validation a/c:", validation_accuracy)


# In[ ]:


#write outputs:
# shot_arr = np.asarray(test['shot_id_number'])
test_rows = data.index[data.is_goal.isnull()]
count = 0
f = open("submissionAP.csv","w+")
print("shot_id_number,is_goal", file=f)
# print("shot_id_number,is_goal", file=f)
for i in range(len(test_rows)):
    print(str(int(test_rows[i]+1))+","+str(prediction[i]), file=f)
    count+=1
print(count)
f.close()


# In[ ]:


print(validation_gtruth)


# In[ ]:


print(len(test))


# In[ ]:


# list(zip(train[features],clf2.feature_importances_))
features_imp = pd.DataFrame(clf.feature_importances_, index=features,columns=['importance']).sort_values('importance', ascending=False)
print(features_imp)


# In[ ]:



print(prediction)
    


# In[ ]:


print(test_rows)

