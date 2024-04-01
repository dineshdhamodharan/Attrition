#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import datasets
import sklearn.metrics as metrics
from sklearn.decomposition import PCA 


from datetime import datetime

import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.model_selection import cross_val_score


#  # Import Data Sets

# In[2]:


df = pd.read_csv('Attrition Extract Model.csv', header =0)
df1 = pd.read_csv('Attrition Extract Model_1.csv', header =0)
df2 = pd.read_csv('Attrition Extract Model_2.csv', header =0)


# # concat Data sets into one file 

# In[3]:


frames = [df,df1,df2]

mdf = pd.concat(frames)


# In[4]:


mdf.head() 


# # Describing and understanding the variables

# In[5]:


mdf.astype('object').describe().transpose()


# In[6]:


mdf.dtypes   


# # Dropping the unwanted variables

# In[7]:


mdf1= mdf.drop(['Behavior Score','BEACON_Score','Months Active Credit Posting 12 months # From Close Date','Months Active Debit Posting 12 months # From Close Date','Sum of Cash Advance Balance 12 months $ From Close Date','pre_durable_key','Count of Account Hashed','Permanent Close Date','Current Credit Line $','Sum Rewards Earned 12 months #','Sum Rewards Redeemed 12 months #','Sum of cash Advance Statement Count 12 months # From Close Date','Sum  Bal Amt 12 mnths $','account_system_entry_date','Avg Bal Amt Q1 $','Avg Bal Amt Q2 $','Avg Bal Amt Q3 $','Avg Bal Amt Q4 $','Sale Count Q1 #','Sale Count Q2 #','Sale Count Q3 #','Sale Count Q4 #','Sale Amount Q1 $','Sale Amount Q2 $','Sale Amount Q3 $','Sale Amount Q4 $'] , axis=1)


# In[8]:


mdf1 = mdf1.rename(columns = {"Months Active Debit OR Credit Posting 12 months # From Close Date": "Months Active during 12 months # From Close Date"})


# In[9]:


mdf1.dtypes  


# In[10]:


mdf1.describe()


# # Checking Missing Values 

# In[11]:


def percentage_of_miss():
  dff=mdf1[mdf1.columns[mdf1.isnull().sum()>=1]]
  total_miss = mdf1.isnull().sum().sort_values(ascending=False)
  percent_miss = (mdf1.isnull().sum()/mdf1.isnull().count()).sort_values(ascending=False)
  missing_Data = pd.concat([total_miss, percent_miss], axis=1, keys=['Number of Missing', 'Percentage'])
  return(missing_Data)


# In[12]:


percentage_of_miss()


# In[13]:


mdf2= mdf1.copy()


# In[14]:


mdf3= mdf2.copy()


# In[15]:


mdf2= mdf2.drop(['Account Hashed'] , axis=1)


# In[16]:


mdf2.head()


# In[17]:


mdf3.head()


# # Filling Missing Values

# In[18]:


mdf2['Permanent Close Reason Name'] = mdf2['Permanent Close Reason Name'].fillna(0)


# In[19]:


mdf2['Permanent Close Reason Name'] = mdf2['Permanent Close Reason Name'].replace('Client Request', 1) 


# In[20]:


mdf2['Product Name'] = pd.factorize(mdf2['Product Name'])[0]


# In[21]:


mdf2 = mdf2.apply (pd.to_numeric, errors='coerce')
mdf2 = mdf2.fillna(0)


# In[22]:


mdf2.head()


# # Describing and understanding the variables

# In[23]:


mdf2.astype('object').describe().transpose()


# In[24]:


mdf2.describe()


# In[25]:


mdf2.dtypes  


# # Checking missing value after Filling values

# In[26]:


def percentage_of_miss():
  dff2=mdf2[mdf2.columns[mdf2.isnull().sum()>=1]]
  total_miss = mdf2.isnull().sum().sort_values(ascending=False)
  percent_miss = (mdf2.isnull().sum()/mdf2.isnull().count()).sort_values(ascending=False)
  missing_Data = pd.concat([total_miss, percent_miss], axis=1, keys=['Number of Missing', 'Percentage'])
  return(missing_Data)


# In[27]:


percentage_of_miss()


# In[28]:


mdf2.apply(lambda x:sum(x.isnull()))


# # correlation Matrix

# In[29]:


corr_ad_Data = mdf2.corr()


# In[30]:


corr_ad_Data


# In[31]:


corr = corr_ad_Data.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.heatmap(corr_ad_Data, vmin = -1, vmax = 1, annot = True, cmap= 'cool')


# In[33]:


dfCorr = mdf2.corr()
filteredDf = dfCorr[((dfCorr >= .5) | (dfCorr <= -.5)) & (dfCorr !=1.000)]
plt.figure(figsize=(30,10))
sns.heatmap(filteredDf, annot=True, cmap="Reds")
plt.show()


# # variance_inflation_factor

# In[34]:


vif =[] # empty list
for i in range(mdf2.shape[1]):
    vif.append(variance_inflation_factor(exog=mdf2.values,exog_idx=i))

pd.DataFrame({'coef name' : mdf2.columns,
             'vif':np.around(vif,3)})


# # Outlier Analysis

# In[35]:


def outlier_analysis(col):
    Q1=mdf2[col].quantile(0.25)
    Q3=mdf2[col].quantile(0.75)
    IQR=Q3-Q1
    UIF=Q3+1.5*(IQR)
    LIF=Q1-1.5*(IQR)
    data_out =mdf2[(mdf2[col]<LIF) | (mdf2[col]>UIF)] 
    sns.distplot(data_out[col])    #Plotting univariate distributions.By default, this will draw a histogram and fit a kernel density estimate (KDE)
   
    return data_out[col] .describe()


# # Yield 

# In[36]:


outlier_analysis('Sum Yield 12 months %')


# In[37]:


mdf2['Sum Yield 12 months %'].min()


# In[38]:


mdf2['Sum Yield 12 months %'].max()


# In[39]:


mdf2['Sum Yield 12 months %'].values[mdf2['Sum Yield 12 months %'].values <= 0] = 0


# In[40]:


mdf2['Sum Yield 12 months %'].values[mdf2['Sum Yield 12 months %'].values > 0.30] = 0.31


# In[41]:


mdf2['Sum Yield 12 months %'].min()


# In[42]:


mdf2['Sum Yield 12 months %'].max()


# In[43]:


sns.boxplot(x=mdf2['Sum Yield 12 months %'])


# # Cash Amount 

# In[44]:


outlier_analysis('Sum of Cash Amount 12 months $ From Close Date')


# In[45]:


mdf2['Sum of Cash Amount 12 months $ From Close Date'].min()


# In[46]:


mdf2['Sum of Cash Amount 12 months $ From Close Date'].max()


# In[47]:


mdf2['Sum of Cash Amount 12 months $ From Close Date'].values[mdf2['Sum of Cash Amount 12 months $ From Close Date'].values > 5000] = 5001


# In[48]:


mdf2['Sum of Cash Amount 12 months $ From Close Date'].min()


# In[49]:


mdf2['Sum of Cash Amount 12 months $ From Close Date'].max()


# In[50]:


sns.boxplot(x=mdf2['Sum of Cash Amount 12 months $ From Close Date'])


# # Balance Amount

# In[51]:


outlier_analysis('Avg Bal Amt 12 mnths $')


# In[52]:


sns.boxplot(x=mdf2['Avg Bal Amt 12 mnths $'])


# In[53]:


mdf2['Avg Bal Amt 12 mnths $'].min()


# In[54]:


mdf2['Avg Bal Amt 12 mnths $'].max()


# In[55]:


mdf2['Avg Bal Amt 12 mnths $'].values[mdf2['Avg Bal Amt 12 mnths $'].values <= 0] = 0


# In[56]:


mdf2['Avg Bal Amt 12 mnths $'].values[mdf2['Avg Bal Amt 12 mnths $'].values > 10000] = 10001


# In[57]:


mdf2['Avg Bal Amt 12 mnths $'].min()


# In[58]:


mdf2['Avg Bal Amt 12 mnths $'].max()


# # Sum Sale Amount 12 months $

# In[59]:


outlier_analysis('Sum Sale Amount 12 months $')


# In[60]:


sns.boxplot(x=mdf2['Sum Sale Amount 12 months $'])


# In[61]:


mdf2['Sum Sale Amount 12 months $'].min()


# In[62]:


mdf2['Sum Sale Amount 12 months $'].max()


# In[63]:


mdf2['Sum Sale Amount 12 months $'].values[mdf2['Sum Sale Amount 12 months $'].values > 50000] = 50001


# In[64]:


mdf2['Sum Sale Amount 12 months $'].min()


# In[65]:


mdf2['Sum Sale Amount 12 months $'].max()


# # Sum of Sale Count 12 months # From Close Date

# In[66]:


outlier_analysis('Sum of Sale Count 12 months # From Close Date')


# In[67]:


sns.boxplot(x=mdf2['Sum of Sale Count 12 months # From Close Date'])


# In[68]:


mdf2['Sum of Sale Count 12 months # From Close Date'].min()


# In[69]:


mdf2['Sum of Sale Count 12 months # From Close Date'].max()


# In[70]:


mdf2['Sum of Sale Count 12 months # From Close Date'].values[mdf2['Sum of Sale Count 12 months # From Close Date'].values > 686] = 686


# In[71]:


mdf2['Sum of Sale Count 12 months # From Close Date'].min()


# In[72]:


mdf2['Sum of Sale Count 12 months # From Close Date'].max()


# # Months Active during 12 months # From Close Date

# outlier_analysis('BEACON_Score')

# sns.boxplot(x=mdf2['BEACON_Score'])

#   mdf2['BEACON_Score'].min()

#   mdf2['BEACON_Score'].max()

# # Zero Beacon Score for 21264 account id 

# j= mdf2['BEACON_Score']
# len([1 for i in j if i == 0])

# # account_age

# In[73]:


outlier_analysis('account_age')


# In[74]:


sns.boxplot(x=mdf2['account_age'])


# In[75]:


mdf2['account_age'].min()


# In[76]:


mdf2['account_age'].max()


# # Close Reason

# In[77]:


Close_Reason = mdf2['Permanent Close Reason Name'].value_counts()
Close_Reason 


# # Understanding Data

# In[78]:


mdf2.astype('object').describe().transpose()


# In[79]:


mdf2.dtypes  


# In[80]:


mdf2.head()


# In[81]:


from scipy.stats import norm 

graph_by_variables = ['Months Active during 12 months # From Close Date','Permanent Close Reason Name','Product Name','account_age','Sum Yield 12 months %','Sum of Cash Amount 12 months $ From Close Date','Avg Bal Amt 12 mnths $','Sum Sale Amount 12 months $','Sum of Sale Count 12 months # From Close Date']
plt.figure(figsize=(15,18))

for i in range(0,9):
    plt.subplot(6,3,i+1)
    sns.distplot(mdf2[graph_by_variables[i]].dropna(),fit=norm)
    plt.title(graph_by_variables[i])

plt.tight_layout()


# In[82]:


mdf2[['Months Active during 12 months # From Close Date','Permanent Close Reason Name','Product Name','account_age','Sum Yield 12 months %','Sum of Cash Amount 12 months $ From Close Date','Avg Bal Amt 12 mnths $','Sum Sale Amount 12 months $','Sum of Sale Count 12 months # From Close Date']].hist(figsize=(10,8))
plt.tight_layout()


# In[83]:


#Perform One Hot Encoding using get_dummies method
mdf2 = pd.get_dummies(mdf2, columns = ['Product Name','account_age','Months Active during 12 months # From Close Date'],
                              drop_first=True)


# In[84]:


mdf2.head()


# In[85]:


#Perform Feature Scaling and One Hot Encoding
from sklearn.preprocessing import StandardScaler

#Perform Feature Scaling on 'tenure', 'MonthlyCharges', 'TotalCharges' in order to bring them on same scale
standardScaler = StandardScaler()
columns_for_ft_scaling = ['Sum Yield 12 months %','Sum of Cash Amount 12 months $ From Close Date','Sum of Cash Amount 12 months $ From Close Date','Sum Sale Amount 12 months $','Sum of Sale Count 12 months # From Close Date']

#Apply the feature scaling operation on dataset using fit_transform() method
mdf2[columns_for_ft_scaling] = standardScaler.fit_transform(mdf2[columns_for_ft_scaling])


# In[86]:


mdf2.head()


# In[87]:


mdf2.astype('object').describe().transpose()


# In[88]:


mdf2.columns


# # Splitting X, Y

# In[89]:


y = mdf2['Permanent Close Reason Name']
X = mdf2.drop(['Permanent Close Reason Name'], axis = 1)


# In[90]:


y.value_counts()


# # Balancing the data using SMOTE

# python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org imbalanced-learn

# In[92]:


from imblearn.over_sampling import SMOTE


# In[93]:


over_sampler = SMOTE(k_neighbors=2)
X_res, y_res = over_sampler.fit_resample(X, y)


# In[94]:


y_res.value_counts()


# # Test Train Split

# In[95]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=0)


# # Scalling the values 

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train_scaled = sc.fit_transform(X_train)
# X_test_scaled = sc.transform(X_test)

# In[96]:


y_train.value_counts()


# In[97]:


y_test.value_counts()


# # Model Building

# #  ------------ Logistic Regression Model -------------

# In[98]:


#Fit the logistic Regression Model
LogModel = LogisticRegression(random_state=50)
LogModel.fit(X_train, y_train)

#Predict the value for new, unseen data
Log_Pred = LogModel.predict(X_test)

# Find Accuracy using accuracy_score method
LogModel_Accuracy =round(metrics.accuracy_score(y_test, Log_Pred)*100, 2)


# # Test score

# In[99]:


LogModel.score(X_test,y_test)


# # Train Score

# In[100]:


LogModel.score(X_train,y_train)


# # Confusion Matrix

# In[101]:


print(confusion_matrix(y_test,Log_Pred))


# # Accuracy Score

# In[102]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, Log_Pred))


# # classification_report

# In[103]:


print(classification_report(y_test,Log_Pred))


# # K_Fold Cross validation

# In[104]:


cross_val_score(LogModel,X,y,cv=5)


# In[105]:


cross_val_score(LogModel,X,y,cv=5).mean()


# #    --------------Decision Tree Classification Model ----------------

# In[106]:


#Fit the Decision Tree Classification Model
from sklearn.tree import DecisionTreeClassifier
dtmodel = DecisionTreeClassifier(criterion = "gini", random_state = 50)
dtmodel.fit(X_train, y_train) 
  
#Predict the value for new, unseen data
dt_pred = dtmodel.predict(X_test)

# Find Accuracy using accuracy_score method
dt_accuracy = round(metrics.accuracy_score(y_test, dt_pred) * 100, 2)


# # Test score

# In[107]:


dtmodel.score(X_test,y_test)


# # Train Score

# In[108]:


dtmodel.score(X_train,y_train)


# # Confusion Matrix

# In[109]:


print(confusion_matrix(y_test,dt_pred))


# # Accuracy Score

# In[110]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, dt_pred))


# # classification_report

# In[111]:


print(classification_report(y_test,dt_pred))


# # K_Fold Cross validation 

# In[112]:


cross_val_score(dtmodel,X,y,cv=5)


# In[113]:


cross_val_score(dtmodel,X,y,cv=5).mean()


# #  -------------- Random Forest Classification Model --------------

# In[114]:


#Fit the Random Forest Classification Model
from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
rfmodel.fit(X_train, y_train) 
  
#Predict the value for new, unseen data
rf_pred = rfmodel.predict(X_test)

# Find Accuracy using accuracy_score method
rf_accuracy = round(metrics.accuracy_score(y_test, rf_pred) * 100, 2)


# # Test score

# In[115]:


rfmodel.score(X_test,y_test)


# # Train Score

# In[116]:


rfmodel.score(X_train,y_train)


# # Confusion Matrix

# In[117]:


print(confusion_matrix(y_test,rf_pred))


# # Accuracy Score

# In[118]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, rf_pred))


# # classification_report

# In[119]:


print(classification_report(y_test,rf_pred))


# # K_Fold Cross validation 

# cross_val_score(rfmodel,X,y,cv=5)

# cross_val_score(rfmodel,X,y,cv=5).mean()

# # Print all the scores at a place 

# In[120]:


# Compare Several models according to their Accuracies
Model_Comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 
              'Decision Tree', 'Random Forest'],
    'Score': [LogModel_Accuracy, 
              dt_accuracy, rf_accuracy]})
Model_Comparison_df = Model_Comparison.sort_values(by='Score', ascending=False)
Model_Comparison_df = Model_Comparison_df.set_index('Score')
Model_Comparison_df.reset_index()


# In[121]:


mdf3.head()


# In[122]:


# Predict the probability of Churn of each customer
mdf2['Probability_of_Churn'] = rfmodel.predict_proba(mdf2[X_test.columns])[:,1]


# mdf2['Probability_of_Churn'] = rfmodel.predict_proba(mdf2[X_test.columns])[:,1]

# In[123]:


# Predicted value stroing into data frame
mdf2['Predicted_value'] = rfmodel.predict(X)


# In[124]:


# Create a Dataframe showcasing probability of Churn of each customer
mdf3['Probablity of churn']= mdf2[['Probability_of_Churn']]


# In[125]:


mdf3['Predicted_value']= mdf2[['Predicted_value']]


# # ------Testing  Start -----

# In[126]:


# Predicted value stroing into data frame
mdf2['Predicted_value_testing']= rfmodel.predict(mdf2[X_test.columns])


# In[127]:


mdf3['Predicted_value_testing'] = mdf2['Predicted_value_testing']


# # ------Testing  End-----

# In[128]:


mdf3.head(50)


# # Exporting Output to CSV

# In[129]:


mdf3.to_csv('Attrition_output.csv', index=False)


# # --------------------------------------------------------------

# # Test Scores

# In[130]:


rfmodel.score(X_test,y_test)


# In[131]:


dtmodel.score(X_test,y_test)


# In[132]:


LogModel.score(X_test,y_test)


# # Train Scores

# In[133]:


rfmodel.score(X_train,y_train)


# In[134]:


dtmodel.score(X_train,y_train)


# In[135]:


LogModel.score(X_train,y_train)


# # Accuracy score

# In[136]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, rf_pred))


# In[137]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, dt_pred))


# In[138]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, Log_Pred))


# # Confusion matrix 

# In[139]:


print(confusion_matrix(y_test,rf_pred))


# In[140]:


print(confusion_matrix(y_test,dt_pred))


# In[141]:


print(confusion_matrix(y_test,Log_Pred))


# In[ ]:




