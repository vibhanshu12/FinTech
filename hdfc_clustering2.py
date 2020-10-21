# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:03:59 2020

@author: Jain Vibhanshu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:09:20 2020

@author: Jain Vibhanshu
"""

#CLUSTERING

import pandas as pd
import numpy as np
c19_20['MANMOD'] = c19_20['POL_MOT_MANUFACTURER_NAME'] +' ' + c19_20['POL_MOT_MODEL_NAME']
c19_202 = c19_20[c19_20['GARAGE_STATE']=='GUJARAT']
c19_202 = c19_202[c19_202['POL_MOT_TYPE_OF_VEHICLE'] == 'Private Car']
a = pd.read_excel(r'C:\Users\jain vibhanshu\Desktop\VJ\Cases\HDFC Ergo\cars.xlsx')
# =============================================================================
# b = a[a['Flag']==1]#Change var to Sedan, Hatchback, Flag for both
# b = b['MANMOD'].tolist()
# b = list(set(list(b)))
# c19_202 = c19_202[c19_202['MANMOD'].isin(b)] 
# =============================================================================
c19_202 = c19_202.merge(a, on='MANMOD', how='left')
c19_202 = c19_202[c19_202['Flag']==1]
#x = c19_202[['POL_MOT_MANUFACTURER_NAME','POL_MOT_MODEL_NAME', 'MANMOD' ]].drop_duplicates()
c19_202 = c19_202[['CLM_REFERENCE_NUM', 'POL_NUM_TXT',
       'CLM_INTIMATION_DATE', 'CLM_DATE_OF_LOSS',
       'CLM_TOTAL_LOSS_FLAG',
       'CALLER_TYPE', 
       'REGISTRATION_AUTHORITY',  'CLM_INBOX_CODE',
       'CLM_LOSS_CITY', 'GARAGE_STATE','Hatch', 'Sedan',
       
       
        'CLM_ASSMT_SUMM_ID',
       
       'CLM_SURVEYOR_APPROVAL_AMT',
        'CLM_GARAGE_ID', 
       'GARAGE_CITY']].drop_duplicates()

c19_202['CLM_ASSMT_SUMM_ID'] = c19_202['CLM_ASSMT_SUMM_ID'].astype(str)
c19_202['RepairFlag'] = np.where(c19_202['CLM_ASSMT_SUMM_ID'].str.contains('PART'),1,0)
df = c19_202.copy()

df['CLM_TOTAL_LOSS_FLAG'] = df['CLM_TOTAL_LOSS_FLAG'].replace( np.nan, 'N')
loss_dict = {'N':0, 'Y':1}
df['CLM_TOTAL_LOSS_FLAG'] = df['CLM_TOTAL_LOSS_FLAG'].map(loss_dict)

# =============================================================================
# #USE THIS
# #x2['perc'] = x2['SUM']/x2.groupby('CLM_REFERENCE_NUM')['SUM'].transform('sum')
# # =============================================================================
# =============================================================================
partcost = df.groupby(by = ['CLM_REFERENCE_NUM', 'RepairFlag'], as_index=False).agg({'CLM_SURVEYOR_APPROVAL_AMT':'sum'})
partcost = partcost.rename(columns={'CLM_SURVEYOR_APPROVAL_AMT':'CumPartLaborCost'})
df = df.merge(partcost, left_on=['CLM_REFERENCE_NUM', 'RepairFlag'], right_on=['CLM_REFERENCE_NUM', 'RepairFlag'], how='left')
# 
# y = x[['CLM_REFERENCE_NUM','RepairFlag','Labor Cost']].drop_duplicates()
# z = y.T.reset_index()
partcost = df.groupby(by = ['CLM_REFERENCE_NUM'], as_index=False).agg({'CLM_SURVEYOR_APPROVAL_AMT':'sum'})
partcost = partcost.rename(columns={'CLM_SURVEYOR_APPROVAL_AMT':'Total Sum'})
df = df.merge(partcost, left_on=['CLM_REFERENCE_NUM'], right_on=['CLM_REFERENCE_NUM'], how='left')
# 
df['Labor Cost'] = np.where(df['RepairFlag'] == 0, df['CumPartLaborCost']/df['Total Sum'], 1-df['CumPartLaborCost']/df['Total Sum'])
df['Part Cost'] = np.where(df['RepairFlag'] == 1, df['CumPartLaborCost']/df['Total Sum'], 1-df['CumPartLaborCost']/df['Total Sum'])
del df['RepairFlag']
del df['CumPartLaborCost']
del df['CLM_SURVEYOR_APPROVAL_AMT']
del df['CLM_ASSMT_SUMM_ID']
df = df.drop_duplicates()
df['CLM_INTIMATION_DATE'] = df['CLM_INTIMATION_DATE'].str[:9]
df['CLM_DATE_OF_LOSS'] = df['CLM_DATE_OF_LOSS'].str[:9]
df['CLM_INTIMATION_DATE'] = pd.to_datetime(df['CLM_INTIMATION_DATE']).dt.date
df['CLM_DATE_OF_LOSS'] = pd.to_datetime(df['CLM_DATE_OF_LOSS']).dt.date

df['intminuslossdate'] = (df['CLM_INTIMATION_DATE']-df['CLM_DATE_OF_LOSS']).dt.days 
del df['CLM_DATE_OF_LOSS']
from math import ceil
decimal_count = 2
df['Labor Cost'] = df['Labor Cost'].astype(float).round(2)
df['Labor Cost'] = df['Labor Cost'].astype(str).str[:5]
df['Labor Cost'] = df['Labor Cost'].astype(float)
df['Part Cost'] = df['Part Cost'].astype(float).round(2)
df['Part Cost'] = df['Part Cost'].astype(str).str[:5]
df['Part Cost'] = df['Part Cost'].astype(float)
df['Total Sum'] = df['Total Sum'].astype(float).round()


df = df.drop_duplicates()
df = df.sort_values(by = ['CLM_REFERENCE_NUM','CLM_TOTAL_LOSS_FLAG'], ascending = False)
df = df.drop_duplicates(subset='CLM_REFERENCE_NUM', keep='first')

df['lossgaragecity'] = np.where(df['CLM_LOSS_CITY']==df['GARAGE_CITY'],1,0)

pol = pd.read_csv(r'C:\Users\jain vibhanshu\Desktop\VJ\Cases\HDFC Ergo\Policy_Data_additional\pol_cont_vj2.csv')
df['POL_NUM_TXT'] = df['POL_NUM_TXT'].astype(str)
df['POL_NUM_TXT'] = np.where(df['POL_NUM_TXT'].str[0]=="'", df['POL_NUM_TXT'].str[1:], df['POL_NUM_TXT'])
df['POL_NUM_TXT'] = df['POL_NUM_TXT'].astype(float)
pol2 = pol[pol['POL_ENDORSEMENT_TYPE']=='Policy']
pol2 = pol2[['POL_NUM_TXT', 'POL_START_DATE', 'POL_END_DATE','MDM_STATE_NAME','CUST_GENDER']].drop_duplicates()
pol2 = pol2.sort_values(by = ['POL_NUM_TXT','POL_END_DATE'], ascending = True)
pol2 = pol2.drop_duplicates(subset = ['POL_NUM_TXT'], keep='last')
pol2['POL_NUM_TXT'] = pol2['POL_NUM_TXT'].astype(float)
#pol2 = pol[['POL_NUM_TXT', 'POL_START_DATE', 'POL_END_DATE','MDM_STATE_NAME','CUST_GENDER', 'FINANCE_LOB']].drop_duplicates()
df2 = df.merge(pol2, left_on='POL_NUM_TXT', right_on='POL_NUM_TXT', how='left')
df2 = df2.rename(columns={'MDM_STATE_NAME':'PolicyState'})
df2['POL_END_DATE'] = pd.to_datetime(df2['POL_END_DATE']).dt.date
df2['polexpint'] = np.where(df2['POL_END_DATE']>df2['CLM_INTIMATION_DATE'],1,0)
df2['POL_START_DATE'] = pd.to_datetime(df2['POL_START_DATE']).dt.date
df2['polstartint'] = np.where(df2['POL_START_DATE']<df2['CLM_INTIMATION_DATE'],1,0)
df2['intminuspolstart'] = (df2['CLM_INTIMATION_DATE'] - df2['POL_START_DATE']).dt.days
df2['polexpminusint'] = (df2['POL_END_DATE'] - df2['CLM_INTIMATION_DATE']).dt.days

test = pol[pol['POL_ENDORSEMENT_TYPE']=='Endorsement']

test=test[['POL_NUM_TXT','POL_ENDORSEMENT_TYPE']].drop_duplicates()
test['endorseflag']=1

df2 = df2.merge(test, on='POL_NUM_TXT', how='left')
df2['endorseflag'] = np.where(df2['endorseflag']==1,1,0)
del df2['POL_ENDORSEMENT_TYPE']

paid = pd.read_csv(r'C:\Users\jain vibhanshu\Desktop\VJ\Cases\HDFC Ergo\paid_conc_tbu.csv')
paid['POL_NUM_TXT'] = paid['POL_NUM_TXT'].astype(str)
paid['POL_NUM_TXT'] = np.where(paid['POL_NUM_TXT'].str[0]=="'", paid['POL_NUM_TXT'].str[1:], paid['POL_NUM_TXT'])
paid  = paid[paid['Reopen/Reissue'].isin(['N','REOPEN','Reopen'])]
paid=paid[paid['FY']=='FY19-20']
paid=paid[['POL_NUM_TXT','INTIMATION_DATE','LOSS']].drop_duplicates()
paid = paid.groupby(by = ['POL_NUM_TXT','INTIMATION_DATE']).agg({'LOSS':'sum'}).reset_index()
#del paiddata['Unnamed: 0']
def get_rolling_count(grp, freq):
    return grp.rolling(freq, on = 'INTIMATION_DATE')['HASHED_ACCOUNT_KEY'].count()

def get_rolling_sum(grp, freq):
    return grp.rolling(freq, on = 'INTIMATION_DATE')['LOSS'].sum()
# =============================================================================
# Apply rolling sum and get in scope txn
# =============================================================================
paid['INTIMATION_DATE'] = pd.to_datetime(paid['INTIMATION_DATE'])
paid['HASHED_ACCOUNT_KEY'] = paid['POL_NUM_TXT'].apply(hash).astype(int)
paid['rolling_count'] = paid.sort_values(by = 'INTIMATION_DATE',ascending = True).groupby('HASHED_ACCOUNT_KEY',as_index=False, group_keys=False).apply(get_rolling_count, '365D')
paid['rolling_sum'] = paid.sort_values(by = 'INTIMATION_DATE',ascending = True).groupby('HASHED_ACCOUNT_KEY',as_index=False, group_keys=False).apply(get_rolling_sum, '365D')

paid = paid.sort_values(by = ['POL_NUM_TXT','INTIMATION_DATE'], ascending = True)
paid = paid.drop_duplicates(subset = ['POL_NUM_TXT'], keep='last')
paid = paid[['POL_NUM_TXT','INTIMATION_DATE','rolling_count', 'rolling_sum']].drop_duplicates()

df2['POL_NUM_TXT'] = df2['POL_NUM_TXT'].astype(float)
paid['POL_NUM_TXT'] = paid['POL_NUM_TXT'].astype(str)
paid = paid[paid['POL_NUM_TXT'].str[0]!='V']
paid['POL_NUM_TXT'] = paid['POL_NUM_TXT'].astype(float)
#df2['POL_NUM_TXT'] = df2['POL_NUM_TXT'].str[:16]
df2 = df2.merge(paid, on='POL_NUM_TXT', how='left')
del df2['INTIMATION_DATE']
pol3 = pol[pol['POL_ENDORSEMENT_TYPE']=='Policy']
pol3=pol3[['POL_NUM_TXT', 'POL_END_DATE','CUMULATIVE_SI']].drop_duplicates()
pol3 = pol3.sort_values(by =['POL_NUM_TXT', 'POL_END_DATE','CUMULATIVE_SI'], ascending = True)
pol3 = pol3.drop_duplicates(subset = ['POL_NUM_TXT'], keep='last')
#pol3['POL_NUM_TXT'] = pol3['POL_NUM_TXT'].astype(str)
df2 = df2.merge(pol3, on = 'POL_NUM_TXT', how='left')
df2['cumuponSI'] = df2['rolling_sum_y']/df2['CUMULATIVE_SI']

z = pd.read_csv(r'C:\Users\jain vibhanshu\Desktop\VJ\Cases\HDFC Ergo\paid_conc_tbu.csv')
z['POL_NUM_TXT'] = z['POL_NUM_TXT'].astype(str)
z['POL_NUM_TXT'] = np.where(z['POL_NUM_TXT'].str[0]=="'", z['POL_NUM_TXT'].str[1:], z['POL_NUM_TXT'])
z  = z[z['Reopen/Reissue'].isin(['N','REOPEN','Reopen'])]
z=z[z['FY']=='FY19-20']
z['TXT_SURVEYOR_CD'] = z['TXT_SURVEYOR_CD'].astype(str)
z = z[z['TXT_SURVEYOR_CD']!='(blank)']
z2 = z.groupby(by='TXT_SURVEYOR_CD')['CLM_REFERENCE_NUMBER'].nunique().reset_index()
z['TXT_SURVEYOR_CD'] = z['TXT_SURVEYOR_CD'].astype(float)
z['TXT_SURVEYOR_CD'] = z['TXT_SURVEYOR_CD'].round()
z2 = z.groupby(by='TXT_SURVEYOR_CD')['CLM_REFERENCE_NUMBER'].nunique().reset_index()
df2['CLM_INBOX_CODE'] = df2['CLM_INBOX_CODE'].astype(float)
df3 = df2.merge(z2, left_on = 'CLM_INBOX_CODE', right_on = 'TXT_SURVEYOR_CD', how='left')
df3['cumuponSI'] = df3['cumuponSI'].round(2)
df3['rolling_sum'] = df3['rolling_sum'].round()
del df3['POL_END_DATE_y']
df3['policygarageloc'] = np.where(df3['GARAGE_STATE']==df3['PolicyState'],1,0)
df3 = df3.rename(columns = {'CLM_REFERENCE_NUMBER':'SurveyorClmCount', 'POL_END_DATE_y':'POL_END_DATE'})

df3['MHPOLICY'] = np.where(df3['PolicyState']=='MAHARASHTRA', 1, 0)

df3['CLM_INTIMATION_DATE'] = pd.to_datetime(df3['CLM_INTIMATION_DATE'])
df3['intimation_year'] = df3['CLM_INTIMATION_DATE'].dt.year
df3['intimation_month'] = df3['CLM_INTIMATION_DATE'].dt.month
del df3['CLM_INTIMATION_DATE']

reg_dict = {'UPLOAD_IMD':'OTHERS','CLAIMS OFFICER':'OTHERS', 'CLAIMS EXAMINER IV':'OTHERS','GRIEVANCE AND FOLLOWUP':'OTHERS',
            'CLAIMS OFFICER MOTOR':'OTHERS','CALL CENTER':'CALL CENTER','CLAIMS_PORTAL':'CLAIMS_PORTAL'}
df3['REGISTRATION_AUTHORITY'] = df3['REGISTRATION_AUTHORITY'].map(reg_dict)
#del df3['MANMOD']
del df3['CLM_GARAGE_ID']
del df3[ 'GARAGE_CITY']
gender_dict = {'FEMALE':0, 'MALE':1}
df3['CUST_GENDER'] = df3['CUST_GENDER'].map(gender_dict)
#del df3['FINANCE_LOB']
del df3['INTIMATION_DATE']
df3 = df3.rename(columns={'rolling_count_y':'rolling_count', 'rolling_sum_y':'rolling_sum'})
df3= df3[['CLM_REFERENCE_NUM', 
       'CLM_TOTAL_LOSS_FLAG',
       'CALLER_TYPE', 'REGISTRATION_AUTHORITY','Hatch', 'Sedan',
       'Total Sum',
       'Labor Cost', 'Part Cost', 'intminuslossdate', 'lossgaragecity',
       'CUST_GENDER',
       'polexpint', 'polstartint', 'intminuspolstart', 'polexpminusint',
       'endorseflag',
       'rolling_count', 'rolling_sum', 'CUMULATIVE_SI', 'cumuponSI',
        'SurveyorClmCount', 'policygarageloc', 'MHPOLICY', 'intimation_year',
       'intimation_month']]
# =============================================================================
# call_dict = {}
# df3['CALLER_TYPE'] = df3['CALLER_TYPE'].map(call_dict)
# =============================================================================
# =============================================================================
# df3['CLM_TOTAL_LOSS_FLAG'] = df3['CLM_TOTAL_LOSS_FLAG'].astype('category')
# df3['lossgaragecity'] = df3['lossgaragecity'].astype('category')
# df3['polstartint'] = df3['polstartint'].astype('category')
# df3['polexpint'] = df3['polexpint'].astype('category')
# df3['intimation_year'] = df3['intimation_year'].astype('category')
# df3['intimation_month'] = df3['intimation_month'].astype('category')
# df3['endorseflag'] = df3['endorseflag'].astype('category')
# df3['CUST_GENDER'] = df3['CUST_GENDER'].astype('category')
# =============================================================================


def count_categorical(df, group_var, df_name):
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    return categorical

def agg_numeric(df, group_var, df_name):
    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    return agg

df3['Labor Cost'] = df['Labor Cost'].astype(float)
df3['Part Cost'] = df['Part Cost'].astype(float)
col = df3.select_dtypes(include=['object','category']).columns.tolist()
df3 = df3.replace([np.inf, -np.inf], np.nan)
col2 = col.remove('CLM_REFERENCE_NUM')

# =============================================================================
# for i in col:
#     if i=='CLM_REFERENCE_NUM': continue
#     else:
# =============================================================================
df4 = pd.get_dummies(df3, columns = col)
df4 = df4.fillna(df4.mean()).fillna(df4.mode().iloc[0])
df4 = df4.set_index('CLM_REFERENCE_NUM')
df4['CALLER_TYPE_DEALER'] = df4['CALLER_TYPE_DEALER'].astype(int)
df4['CALLER_TYPE_INSURED'] = df4['CALLER_TYPE_INSURED'].astype(int)
df4['CALLER_TYPE_OTHERS'] = df4['CALLER_TYPE_OTHERS'].astype(int)

y = ['REGISTRATION_AUTHORITY_CALL CENTER',
       'REGISTRATION_AUTHORITY_CLAIMS_PORTAL', 'REGISTRATION_AUTHORITY_OTHERS']

for i in y:
    df4[i] = df4[i].astype(int)

df5 = df4.copy()
df5 = df5[df5['CLM_TOTAL_LOSS_FLAG']==0]
df5['Total Sum log'] = np.log(df5['Total Sum'])
df5['rolling sum log'] = np.log(df5['rolling_sum'])
df5 = df5[['CLM_TOTAL_LOSS_FLAG', 'Hatch', 'Sedan',  'Labor Cost',
       'Part Cost', 'intminuslossdate', 'lossgaragecity', 'CUST_GENDER',
       'polexpint', 'polstartint', 'intminuspolstart', 'polexpminusint',
       'endorseflag', 'rolling_count', 'CUMULATIVE_SI',
       'cumuponSI', 'SurveyorClmCount', 'policygarageloc', 'MHPOLICY',
       'intimation_year', 'intimation_month', 'CALLER_TYPE_DEALER',
       'CALLER_TYPE_INSURED', 'CALLER_TYPE_OTHERS',
       'REGISTRATION_AUTHORITY_CALL CENTER',
       'REGISTRATION_AUTHORITY_CLAIMS_PORTAL', 'REGISTRATION_AUTHORITY_OTHERS',
       'Total Sum log', 'rolling sum log']]
df5 = df5.replace([np.inf, -np.inf], np.nan)
df5 = df5.fillna(df5.mean()).fillna(df5.mode().iloc[0])

    import matplotlib.pyplot as plt
    %matplotlib inline
    from sklearn.cluster import KMeans
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df5)
    km = KMeans(n_clusters=5, random_state=1, init='k-means++')
    new = data_scaled._get_numeric_data()
    km.fit(data_scaled)
    predict=km.predict(data_scaled)
    df_kmeans = df5.copy(deep=True)
    df_kmeans['Cluster KMeans'] = pd.Series(predict, index=df_kmeans.index)
x = df_kmeans.head(20)



from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4)
gmm.fit(df4)
labels = gmm.predict(df4)
frame = pd.DataFrame(df4)
frame['cluster'] = labels


import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.axhline(y=6, color='r', linestyle='--')
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
ac = cluster.fit_predict(data_scaled)




# statistics of scaled data
# =============================================================================
# pd.DataFrame(data_scaled).describe()
# # defining the kmeans function with initialization as k-means++
# X = df4.iloc[:, [1, 9]].values
# from sklearn.cluster import KMeans
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#     kmeans.fit(X)
#     # inertia method returns wcss for that model
#     wcss.append(kmeans.inertia_)
#     
# import seaborn as sns
# plt.figure(figsize=(10,5))
# sns.lineplot(range(1, 11), wcss,marker='o',color='red')
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()
# 
# kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
# y_kmeans = kmeans.fit_predict(X)
# 
# 
# 
# # fitting the k means algorithm on scaled data
# plt.figure(figsize=(15,7))
# sns.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1',s=50)
# sns.scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2',s=50)
# sns.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'green', label = 'Cluster 3',s=50)
# sns.scatterplot(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], color = 'grey', label = 'Cluster 4',s=50)
# sns.scatterplot(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], color = 'orange', label = 'Cluster 5',s=50)
# sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', 
#                 label = 'Centroids',s=300,marker=',')
# plt.grid(False)
# plt.title('Clusters of claims')
# plt.xlabel('')
# plt.ylabel('')
# plt.legend()
# plt.show()
# 
# =============================================================================


km = KMeans(n_clusters=5, random_state=1, init='k-means++')
new = df4._get_numeric_data()
km.fit(data_scaled)
predict=km.predict(data_scaled)
df_kmeans = df4.copy(deep=True)
df_kmeans['Cluster KMeans'] = pd.Series(predict, index=df_kmeans.index)
x = df_kmeans.head(20)

%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt
Ks = range(1, 25)
km = [KMeans(n_clusters=i, random_state=1) for i in Ks]
my_matrix = df4._get_numeric_data()
score = [km[i].fit(my_matrix).score(my_matrix) for i in range(len(km))]
        
plt.plot(Ks, score)
plt.show()

plt.scatter(X[df_kmeans==0, 0], X[df_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', label ='Cluster 5')



df3.to_csv(r'C:\Users\jain vibhanshu\Desktop\VJ\Cases\HDFC Ergo\cluster2.csv')