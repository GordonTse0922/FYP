# coding: utf-8

# In[1]:


# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import chi2,r_regression
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.model_selection import train_test_split
import yfinance as yf
import talib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import LSTM,Dropout, BatchNormalization, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.models import Sequential
from keras.optimizer_v2.adam import Adam


# In[2]:


data=yf.download('0005.hk',"2017-09-28","2021-09-24")
data.head()


# In[3]:


data['Prediction']=data['Adj Close'].shift(-1)
data.head()


# In[4]:


data['H-L'] = data['High'] - data['Low']
data['O-C'] = data['Close'] - data['Open']
data["% Change"]=data["Close"].shift(1).pct_change()
data['3day MA'] = data['Close'].shift(1).rolling(window = 3).mean()
data['10day MA'] = data['Close'].shift(1).rolling(window = 10).mean()
data['30day MA'] = data['Close'].shift(1).rolling(window = 30).mean()
data['Std_dev']= data['Close'].shift(1).rolling(5).std()
data['RSI'] = talib.RSI(data['Close'].values, timeperiod = 9)
data['Williams %R'] = talib.WILLR(data['High'].values, data['Low'].values, data['Close'].values, 7)
data.head()


# In[5]:


data.dropna(inplace=True)


# In[6]:


# Feature extraction
X=data.drop('Prediction',1)
Y=data['Prediction'].values.reshape(-1,1)
test = SelectKBest(score_func=r_regression, k=5)
fit = test.fit(X, Y)

# Summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)

features = fit.transform(X)
# Summarize selected features
print(features[0:5,:])


# In[7]:


cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model)
fit = rfe.fit(X, Y)
temp = pd.Series(fit.support_,index = cols)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# In[8]:


ridge = Ridge(alpha=1.0)
ridge.fit(X,Y)


# In[9]:


#no of features
nof_list=np.arange(1,16)
high_score=0
#Variable to store the optimum features
nof=0
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,n_features_to_select=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


# In[10]:


cols = list(X.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, n_features_to_select=6)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,Y)
#Fitting the data to model
model.fit(X_rfe,Y)
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# In[11]:


reg = LassoCV()
reg.fit(X, Y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,Y))
coef = pd.Series(reg.coef_, index = X.columns)


# In[12]:


X.columns


# In[13]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[14]:


imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")


# In[15]:


def correlation(dataset,threshold):
    col_corr=set() # set will contains unique values.
    corr_matrix=dataset.corr() #finding the correlation between columns.
    for i in range(len(corr_matrix.columns)): #number of columns
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold: #checking the correlation between columns.
                colName=corr_matrix.columns[i] #getting the column name
                col_corr.add(colName) #adding the correlated column name heigher than threshold value.
    return col_corr #returning set of column names
col=correlation(X,0.8)
print('Correlated columns:',col)


# In[16]:


from mlxtend.feature_selection import SequentialFeatureSelector as sfs
#I am going to use RandomForestRegressor algoritham as an estimator. Your can select other regression alogritham as well.
from sklearn.ensemble import RandomForestRegressor
#k_features=10 (It will get top 10 features best suited for prediction)
#forward=True (Forward feature selection model)
#verbose=2 (It will show details output as shown below.)
#cv=5 (Kfold cross valiation: it will split the training set in 5 set and 4 will be using for training the model and 1 will using as validation)
#n_jobs=-1 (Number of cores it will use for execution.-1 means it will use all the cores of CPU for execution.)
#scoring='r2'(R-squared is a statistical measure of how close the data are to the fitted regression line)
forward_model=sfs(RandomForestRegressor(),k_features=10,forward=True,verbose=2,cv=5,n_jobs=-1,scoring='r2')
forward_model.fit(X_train,y_train.ravel())


# In[17]:


#Get the selected feature index.
forward_model.k_feature_idx_


# In[18]:


#Get the column name for the selected feature.
forward_model.k_feature_names_


# In[19]:


from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestRegressor
#k_features=10 (It will get top 10 features best suited for prediction)
#forward=False (Backward feature selection model)
#verbose=2 (It will show details output as shown below.)
#cv=5 (Kfold cross valiation: it will split the training set in 5 set and 4 will be using for training the model and 1 will using as validation)
#n_jobs=-1 (Number of cores it will use for execution.-1 means it will use all the cores of CPU for execution.)
#scoring='r2'(R-squared is a statistical measure of how close the data are to the fitted regression line)
backwardModel=sfs(RandomForestRegressor(),k_features=10,forward=False,verbose=2,cv=5,n_jobs=-1,scoring='r2')
#We will convert our training data into numpy array. If we will not convert it, model is not able to read some of the column names.
backwardModel.fit(np.array(X_train),y_train.ravel())


# In[20]:


#Get the selected feature index.
backwardModel.k_feature_idx_


# In[21]:


#Get the column name for the selected feature.
X_train.columns[list(backwardModel.k_feature_idx_)]


# In[22]:


from mlxtend.feature_selection import ExhaustiveFeatureSelector as efs
#min_features=1 (minimum number of feature)
#max_features=5 (maximum number of feature)
#n_jobs=-1 (Number of cores it will use for execution.-1 means it will use all the cores of CPU for execution.)
#scoring='r2'(R-squared is a statistical measure of how close the data are to the fitted regression line)
emodel=efs(RandomForestRegressor(),min_features=1,max_features=5,scoring='r2',n_jobs=-1)
#Lets take only 10 features which we got from backward feature selection.
miniData=X_train[X_train.columns[list(backwardModel.k_feature_idx_)]]

emodel.fit(np.array(miniData),y_train.ravel())
#If you see below the model creates 637 feature combinations from 10 features.Thats why its computationally very expensive.


# In[23]:


#Get the selected feature index.
emodel.best_idx_


# In[24]:


#Get the column name for the selected feature.
miniData.columns[list(emodel.best_idx_)]


# In[25]:


emodel=efs(RandomForestRegressor(),min_features=1,max_features=5,scoring='r2',n_jobs=-1)
#Lets take only 10 features which we got from backward feature selection.
miniData_forward=X_train[X_train.columns[list(forward_model.k_feature_idx_)]]

emodel.fit(np.array(miniData_forward),y_train.ravel())
#If you see below the model creates 637 feature comb


# In[26]:


#Get the selected feature index.
emodel.best_idx_


# In[27]:


#Get the column name for the selected feature.
miniData_forward.columns[list(emodel.best_idx_)]


# In[28]:


data_selected=data[miniData_forward.columns[list(emodel.best_idx_)]]


# In[29]:


data_selected['Prediction']=data['Prediction']
data_selected


# In[30]:


# Build the LSTM model
X=data_selected.drop('Prediction',1).values
X = X.reshape(X.shape[0], 1, X.shape[1])
print(X)
y=data_selected['Prediction'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
model = Sequential()
model.add(LSTM(5,input_shape=(1, X.shape[2]), return_sequences=True))
model.add(Dense(1))
model.compile(
  loss="mean_squared_error",
  optimizer='Adam'
)


# In[31]:


model.summary()


# In[ ]:


#Fit model with history to check for overfitting
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=40, min_delta=0.0001)
history = model.fit(X_train,y_train,epochs=100,batch_size=20,validation_data=(X_test,y_test),shuffle=False)


# In[ ]:




