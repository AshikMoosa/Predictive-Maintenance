#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# # Import Data

# In[371]:


#set path for raw data
raw_data_path = os.path.join(os.path.pardir,'data','raw')
train_file_path = os.path.join(raw_data_path, 'train.csv')
test_file_path = os.path.join(raw_data_path, 'test.csv')


# In[372]:


#read data with all parameters
train_df = pd.read_csv(train_file_path, index_col='ord')
test_df = pd.read_csv(test_file_path, index_col='ord')


# In[373]:


# get the type
#type(train_df)


# # Basic Structure

# In[374]:


#train_df.info()
train_df.drop(['TIME','LATITUDE','LONGITUDE','ALTITUDE','VEHICLE_ID','BAROMETRIC_PRESSURE','ENGINE_COOLANT_TEMP','FUEL_LEVEL','AMBIENT_AIR_TEMP','INTAKE_MANIFOLD_PRESSURE','MAF','TERM FUEL TRIM BANK 1','FUEL_ECONOMY','LONG TERM FUEL TRIM BANK 2','FUEL_TYPE','AIR_INTAKE_TEMP','FUEL_PRESSURE','SHORT TERM FUEL TRIM BANK 2','SHORT TERM FUEL TRIM BANK 1','ENGINE_RUNTIME','TIMING_ADVANCE','DTC_NUMBER','TROUBLE_CODES','TIMING_ADVANCE','EQUIV_RATIO','aqui'], axis=1 , inplace=True)
test_df.drop(['TIME','LATITUDE','LONGITUDE','ALTITUDE','VEHICLE_ID','BAROMETRIC_PRESSURE','ENGINE_COOLANT_TEMP','FUEL_LEVEL','AMBIENT_AIR_TEMP','INTAKE_MANIFOLD_PRESSURE','MAF','TERM FUEL TRIM BANK 1','FUEL_ECONOMY','LONG TERM FUEL TRIM BANK 2','FUEL_TYPE','AIR_INTAKE_TEMP','FUEL_PRESSURE','SHORT TERM FUEL TRIM BANK 2','SHORT TERM FUEL TRIM BANK 1','ENGINE_RUNTIME','TIMING_ADVANCE','DTC_NUMBER','TROUBLE_CODES','TIMING_ADVANCE','EQUIV_RATIO','aqui'], axis=1 , inplace=True)


train_df['ENGINE_LOAD'] = train_df['ENGINE_LOAD'].str.replace(',','.')
train_df['THROTTLE_POS'] = train_df['THROTTLE_POS'].str.replace(',','.')


test_df['ENGINE_LOAD'] = test_df['ENGINE_LOAD'].str.replace(',','.')
test_df['THROTTLE_POS'] = test_df['THROTTLE_POS'].str.replace(',','.')


# In[375]:


#concat train & test data for cleaning , axis=0/1 ,0-row concat,1-column concat
df = pd.concat((train_df,test_df),axis=0)
df['ENGINE_LOAD'] = df['ENGINE_LOAD'].str.replace(',','.')
df['THROTTLE_POS'] = df['THROTTLE_POS'].str.replace(',','.')



median_ect = df['THROTTLE_POS'].median()

df.THROTTLE_POS.fillna(median_ect, inplace= True)
median_ect = df['SPEED'].median()

df.SPEED.fillna(median_ect, inplace= True)
median_ect = df['ENGINE_LOAD'].median()

df.ENGINE_LOAD.fillna(median_ect, inplace= True)



median_ect = train_df['THROTTLE_POS'].median()

train_df.THROTTLE_POS.fillna(median_ect, inplace= True)
median_ect = train_df['SPEED'].median()

train_df.SPEED.fillna(median_ect, inplace= True)
median_ect = train_df['ENGINE_LOAD'].median()

train_df.ENGINE_LOAD.fillna(median_ect, inplace= True)



median_ect = test_df['THROTTLE_POS'].median()

test_df.THROTTLE_POS.fillna(median_ect, inplace= True)
median_ect = test_df['SPEED'].median()

test_df.SPEED.fillna(median_ect, inplace= True)
median_ect = test_df['ENGINE_LOAD'].median()

test_df.ENGINE_LOAD.fillna(median_ect, inplace= True)


# In[376]:


#df.info()


# In[377]:


#train_df.info()


# In[378]:


#test_df.info()


# In[379]:


pd.isnull(train_df).sum() > 0


# In[380]:


#pd.isnull(test_df).sum() > 0


# In[381]:


#pd.isnull(df).sum() > 0


# # Data Munging - Working with Missing Values

# ### SPEED FILLING

# In[382]:


#df[df.SPEED.isnull()]


# In[383]:


#df.head(10)


# ## FINDING ENGINE SPEED CHANGE RATE

# In[384]:


#df.SPEED.dtype


# In[385]:


#df.info()


# In[386]:


df["EngineSpeedChangeRate"] = df["ENGINE_RPM"].diff().round(4)


# In[387]:


#df.head(20)


# In[388]:


df['EngineSpeedChangeRate'] = df['EngineSpeedChangeRate'].shift(-1)


# In[389]:


#df.head(10)


# In[390]:


#df.tail(10)


# In[391]:


df.EngineSpeedChangeRate.fillna('1557', inplace= True)


# ## FINDING SPEED CHANGE RATE

# In[392]:


#convert obj to float
df["SPEED"] = pd.to_numeric(df["SPEED"])


# In[393]:


#convert float to int
df["SPEED"]=df["SPEED"].astype(np.int64)


# In[394]:


#df.info()


# In[395]:


df["SpeedChangeRate"] = df["SPEED"].diff().round(4)


# In[396]:


df['SpeedChangeRate'] = df['SpeedChangeRate'].shift(-1)


# In[397]:


df.SpeedChangeRate.fillna('55', inplace= True)


# In[398]:


#df.head(10)


# In[399]:


#df.tail(10)


# ### FINDING THROTTLE CHANGE RATE

# In[400]:


#THROTTLE FILLING NAN
df[df.THROTTLE_POS.isnull()]


# In[401]:


#df.THROTTLE_POS.value_counts()


# In[402]:


#median_ect = df['THROTTLE_POS'].median()
#print (median_ect)


# In[403]:


#df.THROTTLE_POS.fillna('25.1', inplace= True)


# In[404]:


#convert obj to float
df["THROTTLE_POS"] = pd.to_numeric(df["THROTTLE_POS"],downcast='float')


# In[405]:


df["ThrottleChangeRate"] = df["THROTTLE_POS"].diff().round(4)


# In[406]:


df['ThrottleChangeRate'] = df['ThrottleChangeRate'].shift(-1)


# In[407]:


df.ThrottleChangeRate.fillna('31.4', inplace= True)


# In[408]:


#df.tail(10)


# In[409]:


#df.info()


# ### FILLING ENGINE LOAD

# In[410]:


#df[df.ENGINE_LOAD.isnull()]


# In[411]:


#df.info()


# In[412]:


#convert obj to float
df["ENGINE_LOAD"] = pd.to_numeric(df["ENGINE_LOAD"],downcast='float').round(4)


# In[413]:


#convert obj to float
df["ThrottleChangeRate"] = pd.to_numeric(df["ThrottleChangeRate"],downcast='float')


# In[414]:


#convert obj to float
df["EngineSpeedChangeRate"] = pd.to_numeric(df["EngineSpeedChangeRate"])
#convert float to int
df["EngineSpeedChangeRate"]=df["EngineSpeedChangeRate"].astype(np.int64)


# In[415]:


#convert obj to float
df["SpeedChangeRate"] = pd.to_numeric(df["SpeedChangeRate"])
#convert float to int
df["SpeedChangeRate"]=df["SpeedChangeRate"].astype(np.int64)


# In[416]:


#df.info()


# ## FINDING RELATIVE RATIO OF SPEED & ENGINE SPEED

# In[417]:


#Applying formula -  Rcz(t)= cs(t)/220   /   zs(t)/8000
#Simplifying     -   Rcz(t)= 36.3636 * (cs(t)/zs(t))
#dividing cs/zs
df['RelRatioVSES'] = (df['SPEED']/df['ENGINE_RPM']).round(4)

#multiply with 36.3636
df['RelRatioVSES'] = (df['RelRatioVSES']*36.3636).round(4)
#convert obj to float
df["RelRatioVSES"] = pd.to_numeric(df["RelRatioVSES"],downcast='float')


# In[418]:


#df.info()


# ## FINDING RELATIVE RATIO OF THROTTLE & ENGINE SPEED

# In[419]:


#Applying formula -  Rjz(t)= jq'(t)/max(jq'(t))   /   zs'(t)/max(zs'(t))
#Simplifying jq'(t)/zs'(t)  *  max/max
#find max(zs'(t))  -  convert obj to float
df["EngineSpeedChangeRate"] = pd.to_numeric(df["EngineSpeedChangeRate"],downcast='float')
x=df["EngineSpeedChangeRate"].max()

#find max(jq'(t))  -  convert obj to float
df["ThrottleChangeRate"] = pd.to_numeric(df["ThrottleChangeRate"],downcast='float')
y=df["ThrottleChangeRate"].max()

#max/max - x/y
z=x/y

#divide jq'  /  zs'
df['RelRatioTPES'] = (df['ThrottleChangeRate']/df['EngineSpeedChangeRate']).round(4)

#multiply with z
df['RelRatioTPES'] = (df['RelRatioTPES']*z).round(4)

#convert obj to float
df["RelRatioTPES"] = pd.to_numeric(df["RelRatioTPES"],downcast='float')


# In[420]:


median_ect = df['RelRatioTPES'].median()

df.RelRatioTPES.fillna(median_ect, inplace= True)


# In[421]:


#df.info()


# # CREATING A NEW DATAFRAME 

# In[422]:


df2 = df[['RelRatioVSES','RelRatioTPES','ENGINE_LOAD']]


# In[423]:


#df2.head(10)


# ### FINDING WHETHER MAINTENANCE NEEDED OR NOT

# In[424]:


#df2.info()


# In[425]:


#function to create a column MaintenanceReq
def f(row):
    global val
    if row['RelRatioVSES'] >= 0.9 and row['RelRatioVSES'] <= 1.3 :
        if row['RelRatioTPES'] >= 0.9 and row['RelRatioTPES'] <= 1.3 :
            if row['ENGINE_LOAD'] >= 20 and row['ENGINE_LOAD'] <= 50 :
                val = "NO"
    else:
         val = "YES"
    return val
#Creating column MaintenanceReq
df2 = df2.assign(MaintenanceReq=df2.apply(f, axis=1))


# In[426]:


#df2['MaintenanceReq'].value_counts()


# In[427]:


df2=df2.round(4)

df2.ENGINE_LOAD = df2.ENGINE_LOAD.round(4)
#df2.head(20)


# In[428]:


df2=df2.replace([np.inf, -np.inf], np.nan)


# In[429]:


df2.RelRatioTPES.fillna('0.0', inplace= True)
df2["RelRatioTPES"] = pd.to_numeric(df2["RelRatioTPES"],downcast='float')


# In[430]:


#df2.info()


# In[431]:


#df2.describe()


# # ENDGAME

# In[432]:


#Creating Train & Test data from df
df2['is_train']= np.random.uniform(0,1,len(df2)) <= .75
#df2.head(20)


# In[433]:


#creating df with test rows and train rows
train, test = df2[df2['is_train']==True], df2[df2['is_train']==False]
#Show no of test & train observations



# In[434]:


#Creating a list of feature column's name
features = df2.columns[:3]
#view feature
#features


# In[435]:


#Creating target
y = pd.factorize(train['MaintenanceReq'])[0]
z= pd.factorize(test['MaintenanceReq'])[0]


# In[436]:





# In[437]:


#Creating Random Forest Classifier
clf = RandomForestClassifier()
#Training the classifier
#clf.fit(train[features],y)
clf = Pipeline([("scale", StandardScaler()),
               ("clf", RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=2))])


# In[438]:


train[features] = SimpleImputer().fit_transform(train[features])
clf=clf.fit(train[features],y)


# In[439]:


test[features] = SimpleImputer().fit_transform(test[features])
preds=clf.predict(test[features])


# In[440]:


#test['MaintenanceReq'].head()


# In[441]:


#create confusion matrix

pd.crosstab(test['MaintenanceReq'],preds, rownames=['Actual Values'], colnames=['Predicted Values'])
print('')
print('')
print('')
print(pd.crosstab(test['MaintenanceReq'],preds, rownames=['Actual Values'], colnames=['Predicted Values']))

# In[442]:


#check accuracy
def checkAccuracy(clf):
    return accuracy_score(z,preds)
    


# In[443]:


acc=checkAccuracy(clf)

#printing accuracy
print('')
print ("acurracy is: ",acc*100)

#check for maintenance
#df2.MaintenanceReq.value_counts()
values = df2['MaintenanceReq'].value_counts().keys().tolist()
counts = df2['MaintenanceReq'].value_counts().tolist()
if(counts[0]>600):
    print('Maintenance Required')
else:
     print('Maintenance Not Required')

# trial 1

#percent= (df2.loc[df2['MaintenanceReq']=='YES'].count()/(df2.loc[df2['MaintenanceReq']=='NO'].count()+df2.loc[df2['MaintenanceReq']== 'YES'].count()))*100


# trial 2
#percent =  (df2['MaintenanceReq'].count('YES')/(df2['MaintenanceReq'].count('YES')+df2['MaintenanceReq'].count('NO')))*100

#if percent >80:
    #print "Maintenance Reqired"
#else:
    #print "Maintenance Requied"


