# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:45:27 2020

@author: Jain Vibhanshu
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:52:51 2020

@author: Jain Vibhanshu
"""
import glob
import pandas as pd
import numpy as np
import os
import numpy as np
from scipy import stats

# Splitting data into training and testing
from sklearn.model_selection import train_test_split

# Imputing missing values and scaling values
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Machine Learning Models
from sklearn.linear_model import LogisticRegression

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Visualizations
import matplotlib.pyplot as plt
# Inline graphs
%matplotlib inline
import seaborn as sns

#taking care of warnings
import warnings
warnings.filterwarnings("ignore")

#for chi square test
from scipy.stats import chi2_contingency
from scipy.stats import chi2

#for ANOVA test
import statsmodels.api as sm
from statsmodels.formula.api import ols
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

# Display up to 60 columns of a dataframe
pd.set_option('display.max_columns', 60)

# Set default font size
plt.rcParams['font.size'] = 24 

# Internal ipython tool for setting figure size
from IPython.core.pylabtools import figsize

# Font size for seaborn
sns.set(font_scale = 2)
# =============================================================================
# Read Data
# =============================================================================
directoryPath = r'C:\Users\jain vibhanshu\Desktop\VJ\Caselets\1 - Python Ensemble\Problem\Problem\Data\\'
app_data_train = pd.read_csv(directoryPath + 'Application_train.csv')
app_data_test = pd.read_csv(directoryPath + 'Application_test.csv')
bur_bal = pd.read_csv(directoryPath + 'Bureau_balance_data.csv')
bur_data = pd.read_csv(directoryPath + 'Bureau_data.csv')
prev_appl = pd.read_csv(directoryPath + 'Previous_application.csv')
data_dict = pd.read_excel(directoryPath + 'Data_dictionary.xlsx')
cols = 'Unnamed: 0'
def drop_col(df):
    del df[cols]
    return(df)
bur_data = drop_col(bur_data)
prev_appl = drop_col(prev_appl)
from autoplotter import run_app # Importing the autoplotter for GUI Based EDA

import plotly.express as px # Importing plotly express to load dataset
df = px.data.tips() # Getting the   Restaurant data

run_app(df) # Calling the autoplotter.run_app
# =============================================================================
# Change datatypes
# =============================================================================
def return_size(df):
    """Return size of dataframe in gigabytes"""
    return round(sys.getsizeof(df) / 1e9, 2)

def convert_types(df, print_info = False):
    
    original_memory = df.memory_usage().sum()
    
    # Iterate through each column
    for c in df:
        
        # Convert ids and booleans to integers
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(np.int32)
            
        # Convert objects to category
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')
            # Booleans mapped to integers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)
        
        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
            
        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)
        
    new_memory = df.memory_usage().sum()
    
    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
        
    return df
# =============================================================================
# Data Headers and stats
# =============================================================================
x = app_data_train.head()
x = bur_bal.head()
x = app_data_train.describe(percentiles=np.linspace(0,1,11)).T
app_data_train = app_data_train[app_data_train['CNT_FAM_MEMBERS']!=0]
app_data_train['incperfam'] = app_data_train['AMT_INCOME_TOTAL']/app_data_train['CNT_FAM_MEMBERS']
app_data_train['log_income'] = np.log(app_data_train['AMT_INCOME_TOTAL'])
app_data_train['log_credit'] = np.log(app_data_train['AMT_CREDIT'])
app_data_train['pos_days_employed'] = (app_data_train['DAYS_EMPLOYED']*(-1))

app_data_train['log_days_employed'] = np.log((app_data_train['DAYS_EMPLOYED']*(-1)))
df_dist = app_data_train.quantile([0, .1, .2, .3, .4, .5, .6, .7, .9, .99, 1]).T.reset_index()


app_data_train = app_data_train[app_data_train['CODE_GENDER']!='XNA']#remove Unknown Gender
gender_dict = {'F':0, 'M':1}
app_data_train['CODE_GENDER'] = app_data_train['CODE_GENDER'].map(gender_dict)

name_type_dict = {'Unaccompanied':'unaccompanied','Family':'family', 'Spouse, partner':'family',
                  'Children':'family', 'Other_B':'others', 'Other_A':'others', 'Group of people':'others'}
app_data_train['NAME_TYPE_SUITE'] = app_data_train['NAME_TYPE_SUITE'].map(name_type_dict)

inc_dict = {'Working':'working','Commercial associate':'commercial associate',
            'Pensioner':'pensioner','State servant':'state-servant','Unemployed':'others',
            'Student':'others','Businessman':'others','Maternity leave':'others'}
app_data_train['NAME_INCOME_TYPE'] = app_data_train['NAME_INCOME_TYPE'].map(inc_dict)

edu_dict = {'Secondary / secondary special':'secondary','Higher education':'higher-education',
            'Incomplete higher':'others','Lower secondary':'others','Academic degree':'others'}
app_data_train['NAME_EDUCATION_TYPE'] = app_data_train['NAME_EDUCATION_TYPE'].map(edu_dict)

app_data_train = app_data_train[app_data_train['NAME_FAMILY_STATUS']!='Unknown']

house_dict = {'House / apartment':'house-apartment','With parents':'others',
              'Municipal apartment':'others','Rented apartment':'others',
              'Office apartment':'others','Co-op apartment':'others'}
app_data_train['NAME_HOUSING_TYPE'] = app_data_train['NAME_HOUSING_TYPE'].map(house_dict)

occupation_dict = {'Laborers':'laborers','Sales staff':'sales','Core staff':'core-staff',
                   'Managers':'managers','Drivers':'drivers','High skill tech staff':'high skill tech',
                   'Accountants':'accountants','Medicine staff':'med-staff',
                   'Security staff':'sec-staff','Cooking staff':'cooking-staff',
                   'Cleaning staff':'cleaning-staff','Private service staff':'pvt-svc-staff',
                   'Low-skill Laborers':'low-skill-laborers','Waiters/barmen staff':'others',
                   'Secretaries':'others','Realty agents':'others','HR staff':'others',
                   'IT staff':'others'}
app_data_train['OCCUPATION_TYPE'] = app_data_train['OCCUPATION_TYPE'].map(occupation_dict)

org_dict = {'Business Entity Type 3':'business','Self-employed':'self-employed',
            'Other':'others','Medicine':'medicine','Business Entity Type 2':'business',
            'Business Entity Type 1':'business','Government':'govt','School':'school',
            'Trade: type 7':'trade','Kindergarten':'kindergarten','Construction':'construction',
            'Transport: type 4':'transport','Trade: type 3':'trade','Industry: type 9':'industry',
            'Industry: type 3':'industry','Security':'security','Housing':'housing',
            'Industry: type 11':'industry','Military':'military-police','Bank':'finserv',
            'Agriculture':'others','Police':'military-police','Transport: type 2':'transport',
            'Postal':'others','Security Ministries':'security','Trade: type 2':'trade',
            'Restaurant':'others','Industry: type 7':'industry','University':'others',
            'Transport: type 3':'transport','Industry: type 1':'industry',
            'Electricity':'others','Hotel':'others','Industry: type 4':'industry',
            'Trade: type 6':'trade','Insurance':'finserv','Industry: type 5':'industry',
            'Emergency':'others','Telecom':'others','Industry: type 2':'industry',
            'Advertising':'others','Realtor':'others','Culture':'others',
            'Industry: type 12':'industry','Trade: type 1':'trade','Mobile':'others',
            'Legal Services':'others','Cleaning':'others',
            'Transport: type 1':'transport','Industry: type 6':'industry',
            'Industry: type 10':'industry','Religion':'others',
            'Industry: type 13':'industry','Trade: type 4':'trade',
            'Trade: type 5':'trade','Industry: type 8':'industry'}
app_data_train['ORGANIZATION_TYPE'] = app_data_train['ORGANIZATION_TYPE'].map(org_dict)
app_data_train.select_dtypes(include=['object']).head()
# =============================================================================
# loan_dict = {'Revolving loans':0, 'Cash loans':1}
# app_data_train['NAME_CONTRACT_TYPE'] = app_data_train['NAME_CONTRACT_TYPE'].map(loan_dict)
# 
# =============================================================================
# =============================================================================
# Align Test Dataframe
# =============================================================================
app_data_test = app_data_test[app_data_test['CNT_FAM_MEMBERS']!=0]
app_data_test['incperfam'] = app_data_test['AMT_INCOME_TOTAL']/app_data_test['CNT_FAM_MEMBERS']
app_data_test['log_income'] = np.log(app_data_test['AMT_INCOME_TOTAL'])
app_data_test['log_credit'] = np.log(app_data_test['AMT_CREDIT'])
app_data_test['pos_days_employed'] = (app_data_test['DAYS_EMPLOYED']*(-1))

app_data_test['log_days_employed'] = np.log((app_data_test['DAYS_EMPLOYED']*(-1)))
df_dist = app_data_test.quantile([0, .1, .2, .3, .4, .5, .6, .7, .9, .99, 1]).T.reset_index()


app_data_test = app_data_test[app_data_test['CODE_GENDER']!='XNA']#remove Unknown Gender
app_data_test['CODE_GENDER'] = app_data_test['CODE_GENDER'].map(gender_dict)

app_data_test['NAME_TYPE_SUITE'] = app_data_test['NAME_TYPE_SUITE'].map(name_type_dict)

app_data_test['NAME_INCOME_TYPE'] = app_data_test['NAME_INCOME_TYPE'].map(inc_dict)

app_data_test['NAME_EDUCATION_TYPE'] = app_data_test['NAME_EDUCATION_TYPE'].map(edu_dict)

app_data_test = app_data_test[app_data_test['NAME_FAMILY_STATUS']!='Unknown']

app_data_test['NAME_HOUSING_TYPE'] = app_data_test['NAME_HOUSING_TYPE'].map(house_dict)

app_data_test['OCCUPATION_TYPE'] = app_data_test['OCCUPATION_TYPE'].map(occupation_dict)

app_data_test['ORGANIZATION_TYPE'] = app_data_test['ORGANIZATION_TYPE'].map(org_dict)
app_data_test.select_dtypes(include=['object']).head()
# =============================================================================
# loan_dict = {'Revolving loans':0, 'Cash loans':1}
# app_data_test['NAME_CONTRACT_TYPE'] = app_data_test['NAME_CONTRACT_TYPE'].map(loan_dict)
# 
# =============================================================================
# 
# =============================================================================



# Ensure all missing are NaN
app_data_train = app_data_train.fillna(np.nan)
app_data_test = app_data_test.fillna(np.nan)
bur_bal = bur_bal.fillna(np.nan)
bur_data = bur_data.fillna(np.nan)
prev_appl = prev_appl.fillna(np.nan)

# =============================================================================
# Feature Creation
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



bureau_counts = count_categorical(bur_data, group_var = 'SK_ID_CURR', df_name = 'bur_data')
bureau_counts.head()
bureau_agg = agg_numeric(bur_data.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bur_data')
bureau_agg.head()
bureau_balance_counts = count_categorical(bur_bal, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_counts.head()
bureau_balance_agg = agg_numeric(bur_bal, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_agg.head()
# Dataframe grouped by the loan
bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')
bureau_by_loan.head()
# Merge to include the SK_ID_CURR
bureau_by_loan = bur_data[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, on = 'SK_ID_BUREAU', how = 'left')
bureau_by_loan.head()

# Aggregate the stats for each client
bureau_balance_by_cust = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'cust')
bureau_balance_by_cust.head()

previous_agg = agg_numeric(prev_appl, 'SK_ID_CURR', 'previous')
print('Previous application aggregation shape: ', previous_agg.shape)
previous_agg.head()
previous_counts = count_categorical(prev_appl, 'SK_ID_CURR', 'previous')
print('Previous application counts shape: ', previous_counts.shape)
previous_counts.head()


original_features = list(app_data_train.columns)
print('Original Number of Features: ', len(original_features))


col = app_data_train.select_dtypes(include=['object']).columns
app_data_train = pd.get_dummies(app_data_train, columns = col)
app_data_train = app_data_train.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
app_data_train = app_data_train.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')
app_data_train = app_data_train.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')
app_data_train = app_data_train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
app_data_train = app_data_train.merge(bureau_balance_by_cust, on = 'SK_ID_CURR', how = 'left')

app_data_train['debtToCredRatio'] = app_data_train['bur_data_AMT_CREDIT_SUM_DEBT_mean']/app_data_train['bur_data_AMT_CREDIT_SUM_mean']
app_data_train['debtToIncRatio'] = app_data_train['bur_data_AMT_CREDIT_SUM_DEBT_mean']/app_data_train['AMT_INCOME_TOTAL']
app_data_train = app_data_train.replace([np.inf, -np.inf], np.nan)


new_features = list(app_data_train.columns)
print('Number of features using previous loans from other institutions data: ', len(new_features))



# =============================================================================
# missing values 
# =============================================================================
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
missing_train = missing_values_table(app_data_train)
missing_train.head(10)
missing_columns = list(missing_train[missing_train['% of Total Values'] > 50].index)
print('We will remove %d columns from train set.' % len(missing_columns))


#app_data_train = convert_types(app_data_train)

#align the test dataframe columns
# =============================================================================
# app_data_test_agg = agg_numeric(app_data_test, 'SK_ID_CURR', 'test')
# print('Test aggregation shape: ', app_data_test_agg.shape)
# app_data_test_agg.head()
# =============================================================================
col = app_data_test.select_dtypes(include=['object']).columns
app_data_test = pd.get_dummies(app_data_test, columns = col)
app_data_test = app_data_test.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
app_data_test = app_data_test.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')
app_data_test = app_data_test.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')
app_data_test = app_data_test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
app_data_test = app_data_test.merge(bureau_balance_by_cust, on = 'SK_ID_CURR', how = 'left')

app_data_test['debtToCredRatio'] = app_data_test['bur_data_AMT_CREDIT_SUM_DEBT_mean']/app_data_test['bur_data_AMT_CREDIT_SUM_mean']
app_data_test = app_data_test.replace([np.inf, -np.inf], np.nan)
app_data_test['debtToIncRatio'] = app_data_test['bur_data_AMT_CREDIT_SUM_DEBT_mean']/app_data_train['AMT_INCOME_TOTAL']


print('Shape of Testing Data: ', app_data_test.shape)

train_labels = app_data_train['TARGET']

# Align the dataframes, this will remove the 'TARGET' column
train, test = app_data_train.align(app_data_test, join = 'inner', axis = 1)

app_data_train['TARGET'] = train_labels
print('Training Data Shape: ', app_data_train.shape)
print('Testing Data Shape: ', app_data_test.shape)



app_data_train = app_data_train.drop(columns=list(missing_columns))
#app_data_train = app_data_train.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(x.mode()[0], inplace = True), axis = 1)
app_data_train = app_data_train.fillna(app_data_train.mean()).fillna(app_data_train.mode().iloc[0])
df_dist = app_data_train.quantile([0, .1, .2, .3, .4, .5, .6, .7, .9, .99, 1]).T.reset_index()
col_out = ['DAYS_EMPLOYED', 'OWN_CAR_AGE','REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'pos_days_employed']
col_in = app_data_test.select_dtypes(include=['number']).columns

def outlier_dup(df,col_in, col_out):
    for i in col_in:
        if i in col_out: continue
        else:
            median = float(df[i].median())
            quantile = df[i].quantile(0.99)
            df[i] = np.where(df[i] > quantile, median, df[i])
    return(df)

app_data_train = outlier_dup(app_data_train,col_in, col_out)


app_data_test = app_data_test.drop(columns=list(missing_columns))
#app_data_test = app_data_test.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(x.mode()[0], inplace = True), axis = 1)
app_data_test = app_data_test.fillna(app_data_test.mean()).fillna(app_data_test.mode().iloc[0])
df_dist = app_data_test.quantile([0, .1, .2, .3, .4, .5, .6, .7, .9, .99, 1]).T.reset_index()
app_data_test = outlier_dup(app_data_test,col_in,col_out)
# Remove variables to free memory
import gc
gc.enable()
del prev_appl, previous_agg, previous_counts, bureau_counts, bureau_agg, bur_data, bureau_balance_counts, bureau_balance_agg, bur_bal
gc.collect()
# =============================================================================
# 
# =============================================================================
features = [f for f in app_data_train.columns if f not in ['TARGET']]
len(features)
app_data_train[features] = app_data_train[features].fillna(app_data_train[features].mean()).clip(-1e9,1e9)
X = app_data_train[features].values
Y = app_data_train['TARGET'].values.ravel()
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import numpy as np
###initialize Boruta
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, class_weight={0:1,1:11.3}, max_depth=10)
boruta_feature_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=4242, max_iter = 50, perc = 90)
boruta_feature_selector.fit(X, Y)
### print results
X_filtered = boruta_feature_selector.transform(X)
X_filtered.shape
final_features = list()
indexes = np.where(boruta_feature_selector.support_ == True)
for x in np.nditer(indexes):
    final_features.append(features[x])
print(final_features)
X = app_data_train[final_features]
y = app_data_train['TARGET']
# =============================================================================
# SMOTE
# =============================================================================
X = app_data_train.loc[:, app_data_train.columns != 'TARGET']
y = app_data_train.loc[:, app_data_train.columns == 'TARGET']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['TARGET'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['TARGET']==0]))
print("Number of subscription",len(os_data_y[os_data_y['TARGET']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['TARGET']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['TARGET']==1])/len(os_data_X))
X = os_data_X
y = os_data_y
# =============================================================================
# RFE
# =============================================================================
data_final_vars=app_data_train.columns.tolist()
X = app_data_train[final_features]
Y = app_data_train['TARGET'].values.ravel()
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(RandomForestClassifier(),50)
rfe = rfe.fit(X, Y)
print(rfe.support_)
print(rfe.ranking_)
f = rfe.get_support(1) #the most important features
X = app_data_train[app_data_train.columns[f]] # final features`
X = os_data_X[os_data_X.columns[f]]
y= os_data_y['TARGET']
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
# create pipeline
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=25)
model = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# =============================================================================
# Feature selection
# =============================================================================
corrs = X.corr()
corrs = corrs.sort_values('TARGET', ascending = False)

# Ten most positive correlations
pd.DataFrame(corrs['TARGET'].head(10))

from collections import defaultdict
threshold = 0.8


flag = False

corr_dict = defaultdict(list)

for row in corrs.index:
    for col in corrs.columns:
        if (col!=row) and (abs(corrs.loc[row, col]) >= threshold):
            flag = True
            corr_dict[row].append(col)

if flag:
    print('High correlation present!')
    print(corr_dict)
else:
    print('No high correlation present!')
    

# Empty dictionary to hold correlated variables
above_threshold_vars = {}

# For each column, record the variables that are above the threshold
for col in corrs:
    above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])

cols_to_remove = []
cols_seen = []
cols_to_remove_pair = []

# Iterate through columns and correlated columns
for key, value in above_threshold_vars.items():
    # Keep track of columns already examined
    cols_seen.append(key)
    for x in value:
        if x == key:
            next
        else:
            # Only want to remove one in a pair
            if x not in cols_seen:
                cols_to_remove.append(x)
                cols_to_remove_pair.append(key)
            
cols_to_remove = list(set(cols_to_remove))
print('Number of columns to remove: ', len(cols_to_remove))
X = X.drop(columns = cols_to_remove)
test_corrs_removed = app_data_test.drop(columns = cols_to_remove)

print('Training Corrs Removed Shape: ', train_corrs_removed.shape)
print('Testing Corrs Removed Shape: ', test_corrs_removed.shape)

# =============================================================================
# def correlation(dataset, threshold):
#     col_corr = set() # Set of all the names of deleted columns
#     corr_matrix = dataset.corr()
#     for i in range(len(corr_matrix.columns)):
#         for j in range(i):
#             if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
#                 colname = corr_matrix.columns[i] # getting the name of column
#                 col_corr.add(colname)
#                 print(col_corr)
#                 if colname in dataset.columns:
#                     del dataset[colname] # deleting the column from the dataset
# correlation(yo, 0.8)
# =============================================================================
from statsmodels.stats.outliers_influence import variance_inflation_factor    

#define function to calculate VIF for all columns

def calculate_vif_(df, thresh=3):
    """
    The function calculates VIF for all columns, then drops the column with highest VIF above the threshold.
    The process is then repeated till there are no columns with VIF above the threshold.
    Function outputs the list of columns to be dropped.
    The columns are NOT actually dropped from the input dataframe
    Input Dataframe should only have numeric columns
    """
    variables = list(range(df.shape[1]))
    drop_cols = list()
    dropped = True
    
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(df.iloc[:, variables].values, ix) for ix in range(df.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print(df.iloc[:, variables].columns[maxloc])
            drop_cols.append(df.iloc[:, variables].columns[maxloc])
            del variables[maxloc]
            dropped = True

    print('---')
    print('Columns to be dropped:')
    print(drop_cols)
    return drop_cols
    
drop_cols= calculate_vif_(X._get_numeric_data())
drop_cols2 = [e for e in drop_cols if e not in ('AMT_CREDIT','AMT_INCOME_TOTAL')]
train_corrs_removed.drop(drop_cols2,axis=1,inplace=True)

# =============================================================================
# ########################################
# 
# def calculate_woe_iv(dataset, feature, target):
#     lst = []
#     for i in range(dataset[feature].nunique()):
#         val = list(dataset[feature].unique())[i]
#         lst.append({
#             'Value': val,
#             'All': dataset[dataset[feature] == val].count()[feature],
#             'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
#             'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
#         })
#         
#     dset = pd.DataFrame(lst)
#     dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
#     dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
#     dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
#     dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
#     dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
#     iv = dset['IV'].sum()
#     
#     dset = dset.sort_values(by='WoE')
#     
#     return dset, iv
# for col in yo.columns:
#     if col == 'TARGET': continue
#     else:
#         print('WoE and IV for column: {}'.format(col))
#         df, iv = calculate_woe_iv(yo, col, 'TARGET')
#         print(df)
#         print('IV score: {:.2f}'.format(iv))
#         print('\n')
# =============================================================================
        
# =============================================================================
# 
# =============================================================================


def iv_woe(data, target, bins=10, show_woe=False):
    
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    #Extract Column Names
    cols = data.columns
    
    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF

iv, woe = iv_woe(data = X, target = 'TARGET', bins=10, show_woe = True)
print(iv)
print(woe)
iv = iv.sort_values('IV', ascending = False)
x = iv[iv['IV']<0.0099]
drop_cols3 = [e for e in x.Variable.tolist()]
train_corrs_removed.drop(drop_cols3,axis=1,inplace=True)
test_corrs_removed.drop(drop_cols3,axis=1,inplace=True)
xx = train_corrs_removed._get_numeric_data().columns.tolist()
train_corrs_removed.hist(bins=30, figsize=(50, 25))

%matplotlib inline
pd.crosstab(app_data_train.ORGANIZATION_TYPE,app_data_train['TARGET']).plot(kind='bar')
plt.title('AMT_CRED')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('z')
# =============================================================================

#  =============================================================================
# =============================================================================
# Feature Transformation
# =============================================================================
# Train-test-Validation Split

# Separate the dataset into Features and Target variables  

# =============================================================================
# X = train_corrs_removed.loc[:, train_corrs_removed.columns != 'TARGET']
# y = train_corrs_removed['TARGET']
# =============================================================================
# Perform split for train:test ~ 70:30, here train contains (train + validation) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print ("Train dataset: {0}{1}".format(X_train.shape, y_train.shape))
print ("Test dataset: {0}{1}".format(X_test.shape, y_test.shape))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=False, with_std=False)

scale_cols = X_train.columns

# Fit on Training data and then apply it to every feature set present
X_train[scale_cols] = sc.fit_transform(X_train[scale_cols])
X_test[scale_cols] = sc.transform(X_test[scale_cols])
#import statsmodels.api as sm
# Model Building - The most simple approach

# Create an instance of LogisticRegression()
lr = LogisticRegression(class_weight = {0:1,1:11.3},C=0.001,penalty='l1', solver ='liblinear')#after validation & tuning
lr = LogisticRegression(class_weight = {0:1,1:11.3},C=4.90001)#after validation & tuning

lr.fit(X_train, y_train)
# Computing prediction on 'X_test' test dataset, outputs predicted labels
y_pred = lr.predict(X_test)
y_pred_proba = lr.predict_proba(X_test)[:,1]

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss,roc_auc_score
from sklearn.metrics import average_precision_score

test_accuracy = accuracy_score(y_test, y_pred_2)*100
test_auc_roc = roc_auc_score(y_test, y_pred_proba)*100
test_avg_precision = average_precision_score(y_test, y_pred_2)
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred_2))
print('Testing accuracy: %.4f %%' % test_accuracy)
print('Testing AUC: %.4f %%' % test_auc_roc)
print('Testing Average Precision Score: %.4f %%' % test_avg_precision)


[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)

y_pred_train_lr = lr.predict(X_train)
y_pred_proba_train_lr = lr.predict_proba(X_train)[:,1]
train_accuracy = accuracy_score(y_train, y_pred_train_lr)*100
train_auc_roc = roc_auc_score(y_train, y_pred_proba_train_lr)*100
train_avg_precision = average_precision_score(y_train, y_pred_train_lr)
print('Confusion matrix:\n', confusion_matrix(y_train, y_pred_train_lr))
print('Training accuracy: %.4f %%' % train_accuracy)
print('Training AUC: %.4f %%' % train_auc_roc)
print('Training Average Precision Score: %.4f %%' % train_avg_precision)


# Evaluating the model on 'accuracy' metric with 'accuracy_score()' method

print("Accuracy Score: {}".format(accuracy_score(y_test, y_pred)))
print(lr.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(lr.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))
# Let's test the predicted output value of our model for first row example in the test dataset
# Test dataset row 0

#please note, the values are rescaled
X_test.iloc[[0]]
print("Test dataset row 0: Actual Value: {}".format(y_test.values[0]))

print("Test dataset row 0: Predicted Output by the model: {}".format(y_pred[0]))

# =============================================================================
# Model Tuning using GridSearch
# =============================================================================
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

#Define simple model
C = np.arange(1e-05, 5.5, 0.1)
scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}
log_reg = LogisticRegression()

#Simple pre-processing estimators
std_scale = StandardScaler(with_mean=False, with_std=False)
#std_scale = StandardScaler()

#Defining the CV method: Using the Repeated Stratified K Fold

n_folds=5
n_repeats=5

rskfold = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=2)

#Creating simple pipeline and defining the gridsearch
###############################################################################

log_clf_pipe = Pipeline(steps=[('scale',std_scale), ('clf',log_reg)])
log_clf = GridSearchCV(estimator=log_clf_pipe, cv=rskfold,
              scoring=scoring, return_train_score=True,
              param_grid=dict(clf__C=C), refit='Accuracy')

log_clf.fit(X, y)
results = log_clf.cv_results_

print('='*20)
print("best params: " + str(log_clf.best_estimator_))
print("best params: " + str(log_clf.best_params_))
print('best score:', log_clf.best_score_)
print('='*20)

plt.figure(figsize=(10, 10))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",fontsize=16)

plt.xlabel("Inverse of regularization strength: C")
plt.ylabel("Score")
plt.grid()

ax = plt.axes()
ax.set_xlim(0, C.max()) 
ax.set_ylim(0.35, 0.95)
X_axis = np.array(results['param_clf__C'].data, dtype=float)

for scorer, color in zip(list(scoring.keys()), ['g', 'k', 'b']): 
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = -results['mean_%s_%s' % (sample, scorer)] if scoring[scorer]=='neg_log_loss' else results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = -results['mean_test_%s' % scorer][best_index] if scoring[scorer]=='neg_log_loss' else results['mean_test_%s' % scorer][best_index]
        
    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.show()



# =============================================================================
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
# logreg = LogisticRegression(penalty='l1', C=0.001,solver ='liblinear')
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
# from sklearn.metrics import accuracy_score
# print("Accuracy Score: {}".format(accuracy_score(y_test, y_pred)))
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))
# print("Test dataset row 0: Actual Value: {}".format(y_test.values[0]))
# 
# print("Test dataset row 0: Predicted Output by the model: {}".format(y_pred[0]))
# import statsmodels.api as sm
# =============================================================================
import statsmodels.api as sm_api
X_train = sm_api.add_constant(X_train)
X_train.head()
X_test = sm_api.add_constant(X_test)
model = sm.Logit(y_train, X_train)
result = model.fit()
result_summary = result.summary()
results_as_html = result_summary.tables[1].as_html()
x=pd.read_html(results_as_html, header=0, index_col=0)[0]
# =============================================================================
# Model Statistics
# =============================================================================
from bisect import bisect_left, bisect_right

def concordance_discordance(zeros_df, ones_df):
    """
    The following function calculates concordance, discordance, ties and Somers-D
    Two inputs
    1. Dataframe with predicted probability for all actual 1s
    2. Dataframe with predicted probability for all actual 0s
    """
    
    zeros_idx = zeros_df.copy().reset_index()
    ones_idx = ones_df.copy().reset_index()

    zeros = zeros_idx.drop('index', axis=1)
    ones = ones_idx.drop('index', axis=1)
    
    zeros_list = sorted([zeros.iloc[j,1] for j in zeros.index])
    zeros_length = len(zeros_list)
    
    disc = 0
    ties = 0
    conc = 0
    
    for i in ones.index:
        cur_conc = bisect_left(zeros_list, ones.iloc[i,1])
        cur_ties = bisect_right(zeros_list, ones.iloc[i,1]) - cur_conc
        conc += cur_conc
        ties += cur_ties
        
    pairs_tested = zeros_length * len(ones.index)
    disc = pairs_tested - conc - ties
    
    print('Pairs = ', pairs_tested)
    print('Conc = ', conc)
    print('Disc = ', disc)
    print('Tied = ', ties)
    
    concordance = round(conc*100/pairs_tested,4)
    discordance = round(disc*100/pairs_tested,4)
    ties_perc = round(ties*100/pairs_tested,4)
    Somers_D = round((conc - disc)/pairs_tested,4)
    
    print('Concordance = ', concordance, '%')
    print('Discordance = ', discordance, '%')
    print('Tied = ', ties_perc, '%')
    print('Somers D = ', Somers_D)
    
# Predicted values
y_pred_orig = model.predict(params=result.params)
y_pred_orig
prob_df = pd.DataFrame({'Actual Value': y_train, 'Predicted Probability': y_pred_orig})

zeros_df = prob_df[prob_df['Actual Value']==0]
ones_df = prob_df[prob_df['Actual Value']==1]

# Call the function
concordance_discordance(zeros_df, ones_df)

#Deleting variables with high p-value
p_values = pd.DataFrame(result.pvalues).reset_index()
p_values = p_values.rename(columns={'index': 'Features', 0: 'p-value'})

# What features have p-values greater than 0.05 - remove them
alpha = 0.05

# Create a list of dropped columns from p-value
drop_cols_pval = list(p_values[p_values['p-value'] > alpha]['Features'])

dropped = drop_cols_pval

if(len(dropped)!=0):
    print("Dropping: {}".format(dropped))
else:
    print("No Dropping!")
drop_cols_pval = ['const', 'train_NAME_INCOME_TYPE_pensioner_count_norm', 'previous_NAME_CONTRACT_TYPE_Cash loans_count_norm', 'previous_NAME_CONTRACT_TYPE_Consumer loans_count_norm', 'previous_NAME_CONTRACT_TYPE_Revolving loans_count_norm', 'previous_CHANNEL_TYPE_AP+ (Cash loan)_count', 'previous_AMT_CREDIT_count', 'previous_AMT_CREDIT_min', 'bur_data_CREDIT_TYPE_Microloan_count', 'bur_data_AMT_CREDIT_SUM_min', 'bur_data_AMT_CREDIT_SUM_DEBT_max', 'bur_data_AMT_CREDIT_SUM_DEBT_min']
#'train_NAME_INCOME_TYPE_pensioner_count_norm', 'previous_NAME_CONTRACT_TYPE_Cash loans_count_norm', 'previous_NAME_CONTRACT_TYPE_Consumer loans_count_norm', 'previous_NAME_CONTRACT_TYPE_Revolving loans_count_norm', 'previous_CHANNEL_TYPE_AP+ (Cash loan)_count', 'previous_AMT_CREDIT_count', 'previous_AMT_CREDIT_min', 'bur_data_CREDIT_TYPE_Microloan_count', 'bur_data_AMT_CREDIT_SUM_min', 'bur_data_AMT_CREDIT_SUM_DEBT_max', 'bur_data_AMT_CREDIT_SUM_DEBT_min']
#'AMT_CREDIT', 'train_NAME_INCOME_TYPE_pensioner_count_norm', 'previous_NAME_CONTRACT_TYPE_Cash loans_count_norm', 'previous_NAME_CONTRACT_TYPE_Consumer loans_count_norm', 'previous_NAME_CONTRACT_TYPE_Revolving loans_count_norm', 'previous_CHANNEL_TYPE_AP+ (Cash loan)_count', 'previous_AMT_CREDIT_count', 'previous_AMT_CREDIT_min', 'bur_data_CREDIT_TYPE_Microloan_count', 'bur_data_AMT_CREDIT_SUM_min', 'bur_data_AMT_CREDIT_SUM_DEBT_max', 'bur_data_AMT_CREDIT_SUM_DEBT_min', 'debtToCredRatio']
#'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'train_NAME_CONTRACT_TYPE_Cash loans_count', 'train_NAME_CONTRACT_TYPE_Revolving loans_count', 'train_NAME_INCOME_TYPE_commercial associate_count', 'train_NAME_INCOME_TYPE_pensioner_count', 'train_NAME_INCOME_TYPE_state-servant_count', 'train_NAME_INCOME_TYPE_working_count', 'train_NAME_EDUCATION_TYPE_higher-education_count', 'train_NAME_EDUCATION_TYPE_others_count', 'train_NAME_EDUCATION_TYPE_secondary_count', 'train_NAME_FAMILY_STATUS_Single / not married_count', 'train_OCCUPATION_TYPE_sales_count', 'previous_NAME_CONTRACT_TYPE_Cash loans_count', 'previous_NAME_CONTRACT_STATUS_Approved_count_norm', 'previous_NAME_CONTRACT_STATUS_Refused_count', 'previous_CHANNEL_TYPE_AP+ (Cash loan)_count', 'previous_CHANNEL_TYPE_Credit and cash offices_count', 'previous_CHANNEL_TYPE_Credit and cash offices_count_norm', 'previous_CHANNEL_TYPE_Stone_count', 'previous_CHANNEL_TYPE_Stone_count_norm', 'previous_AMT_CREDIT_min', 'previous_AMT_CREDIT_sum', 'bur_data_CREDIT_TYPE_Car loan_count_norm', 'bur_data_CREDIT_TYPE_Consumer credit_count_norm', 'bur_data_CREDIT_TYPE_Credit card_count', 'bur_data_CREDIT_TYPE_Credit card_count_norm', 'bur_data_CREDIT_TYPE_Microloan_count', 'bur_data_CREDIT_TYPE_Mortgage_count_norm', 'bur_data_AMT_CREDIT_SUM_min', 'bur_data_AMT_CREDIT_SUM_DEBT_mean', 'bur_data_AMT_CREDIT_SUM_DEBT_max', 'bur_data_AMT_CREDIT_SUM_DEBT_min', 'cust_bureau_balance_MONTHS_BALANCE_mean_sum', 'cust_bureau_balance_STATUS_X_count_norm_sum', 'debtToCredRatio']
#'DEF_60_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'pos_days_employed', 'train_NAME_CONTRACT_TYPE_Cash loans_count', 'train_NAME_CONTRACT_TYPE_Revolving loans_count', 'train_NAME_INCOME_TYPE_pensioner_count', 'train_NAME_HOUSING_TYPE_house-apartment_count', 'train_NAME_HOUSING_TYPE_others_count', 'previous_NAME_CONTRACT_TYPE_Consumer loans_count_norm', 'previous_NAME_CONTRACT_TYPE_Revolving loans_count_norm', 'previous_NAME_CONTRACT_STATUS_Approved_count_norm', 'previous_CHANNEL_TYPE_AP+ (Cash loan)_count', 'previous_CNT_PAYMENT_mean', 'bur_data_CREDIT_TYPE_Credit card_count_norm', 'bur_data_CREDIT_TYPE_Microloan_count', 'bur_data_AMT_CREDIT_SUM_DEBT_mean', 'bur_data_AMT_CREDIT_SUM_DEBT_max', 'bur_data_AMT_CREDIT_SUM_DEBT_min', 'debtToCredRatio']
X_train2 = X_train.drop(columns=drop_cols_pval, axis=1)
X_test2 = X_test.drop(columns=drop_cols_pval, axis=1)
model2 = sm.Logit(y_train, X_train2)
result2 = model2.fit()
a = result2.summary()
results_as_html = a.tables[1].as_html()
x=pd.read_html(results_as_html, header=0, index_col=0)[0]
p_values2 = pd.DataFrame(result2.pvalues).reset_index()
p_values2 = p_values2.rename(columns={'index': 'Features', 0: 'p-value'})

# What features have p-values greater than 0.05 - remove them
alpha = 0.05

# Create a list of dropped columns from p-value
drop_cols_pval2 = list(p_values2[p_values2['p-value'] > alpha]['Features'])

dropped2 = drop_cols_pval2

if(len(dropped2)!=0):
    print("Dropping: {}".format(dropped2))
else:
    print("No Dropping!")
drop_cols_pval2 = ['AMT_CREDIT', 'bur_data_CREDIT_TYPE_Credit card_count_norm', 'debtToCredRatio']
#'previous_NAME_CONTRACT_TYPE_Cash loans_count_norm', 'previous_NAME_CONTRACT_TYPE_Consumer loans_count_norm', 'previous_NAME_CONTRACT_TYPE_Revolving loans_count_norm','train_NAME_HOUSING_TYPE_house-apartment_count', 'train_NAME_HOUSING_TYPE_others_count']
X_train3 = X_train2.drop(columns=drop_cols_pval2, axis=1)
X_test3 = X_test2.drop(columns=drop_cols_pval2, axis=1)
model3 = sm.Logit(y_train, X_train3)
result3 = model3.fit()
a = result3.summary()
results_as_html = a.tables[1].as_html()
x=pd.read_html(results_as_html, header=0, index_col=0)[0]
p_values3 = pd.DataFrame(result3.pvalues).reset_index()
p_values3 = p_values3.rename(columns={'index': 'Features', 0: 'p-value'})

# What features have p-values greater than 0.05 - remove them
alpha = 0.05

# Create a list of dropped columns from p-value
drop_cols_pval3 = list(p_values3[p_values3['p-value'] > alpha]['Features'])

dropped3 = drop_cols_pval3

if(len(dropped3)!=0):
    print("Dropping: {}".format(dropped3))
else:
    print("No Dropping!")

y_pred3 = model3.predict(params=result3.params)
y_pred3
prob_df = pd.DataFrame({'Actual Value': y_train, 'Predicted Probability': y_pred3})

zeros_df = prob_df[prob_df['Actual Value']==0]
ones_df = prob_df[prob_df['Actual Value']==1]

# Call the function
concordance_discordance(zeros_df, ones_df)

# =============================================================================
# Model Validation
# =============================================================================
from sklearn.model_selection import cross_val_score

# Estimating the accuracy of our model by splitting the data, fitting a model and computing the score 10 
# consecutive times (with different splits each time):

log_reg = LogisticRegression()
scores = cross_val_score(log_reg, X_train3, y_train, cv=5, scoring='f1')
scores

#logreg = LogisticRegression()

# =============================================================================
# Model Fit STats
# =============================================================================

f, ax = plt.subplots(figsize=(9, 6))
_ = plt.plot(fpr, tpr, [0,1], [0, 1])
_ = plt.title('AUC ROC')
_ = plt.xlabel('False positive rate')
_ = plt.ylabel('True positive rate')
plt.style.use('seaborn')
plt.savefig('auc_roc.png', dpi=600)

from numpy import argmax
from numpy import sqrt
gmeans = sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thr[ix], gmeans[ix]))
optimal_idx = np.argmin(np.abs(tpr - fpr)) 
optimal_threshold = thr[optimal_idx]
y_pred_2 = (y_pred_proba > 0.499951 )*1
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred_2))
print(classification_report(y_test, y_pred_2, digits=6))
test_auc_roc = roc_auc_score(y_test, y_pred_2)*100
print('Test AUC: %.4f %%' % test_auc_roc)
from scipy.stats import ks_2samp
#ks_2samp(df.loc[df.y==0,"p"], df.loc[df.y==1,"p"])





# =============================================================================
# y_pred_final = model3.predict(params=result3.params, exog=X_test3)
# y_pred_final
# y_pred_labels = (y_pred_final>0.1).astype(int)
# y_pred_labels
# from sklearn.metrics import confusion_matrix
# print("Confusion Matrix")
# print(confusion_matrix(y_test, y_pred_2))
# sns.heatmap(confusion_matrix(y_test, y_pred_proba_2), annot=True, cmap='Blues', fmt='g')
# plt.ylabel('Actual Label')
# _ = plt.xlabel('Predicted Label')
# =============================================================================

conf_mat = confusion_matrix(y_test, y_pred_2)
print("% of False Positive: {}".format(conf_mat[0][1]*100/(conf_mat[0][0] + conf_mat[0][1] + conf_mat[1][0] + conf_mat[1][1])))

print("% of False Negative: {}".format(conf_mat[1][0]*100/(conf_mat[0][0] + conf_mat[0][1] + conf_mat[1][0] + conf_mat[1][1])))

#KS
ks_df = pd.DataFrame(data={'Actual Value': y_test,
                                 'Predicted Value': y_pred_2,
                                 'Probability': y_pred_proba})
ks_df.head()
ks_df.reset_index(inplace=True)
ks_df.drop(columns='index', axis=1, inplace=True)

ks_df.sort_values(by='Probability', ascending=False, inplace=True)

rows = []

# For loop for computing on deciles
for group in np.array_split(ks_df, 10):
    events = group[group['Actual Value']==1].count()['Actual Value']
    rows.append({'NumCases': len(group), 'NumResponses': events})

ks = pd.DataFrame(rows)

# No. of Non Responses
ks['NumNonResponses'] = ks['NumCases'] - ks['NumResponses']

# %age of Response
ks['PercentResponse'] = round((ks['NumResponses']/ks['NumResponses'].sum())*100, 2)

# %age of Non Response
ks['PercentNonResponse'] = round((ks['NumNonResponses']/ks['NumNonResponses'].sum())*100, 2)

# Cumulative %age Response
ks['CumPercentResponse'] = round((ks['NumResponses'].cumsum()/ks['NumResponses'].sum())*100, 2)

# Cumulative %age Non Response
ks['CumPercentNonResponse'] = round((ks['NumNonResponses'].cumsum()/ks['NumNonResponses'].sum())*100, 2)

# Difference values: KS = Cumulative % Response - Cumulative % Non Response
ks['KS_Value'] = round(ks['CumPercentResponse'] - ks['CumPercentNonResponse'], 2)

# %age Dataset: decile frequency distribution
ks['PercentDataset'] = round((ks['NumCases'].cumsum()/ks['NumCases'].sum())*100, 2)

# KS-statistic value: max(difference/KS values)
ks_val = ks['KS_Value'].max()

# Index of KS-statistic in column
ks_val_index = ks['KS_Value'].idxmax()

print('KS-Statistic Value: {}'.format(ks_val))

# Plot KS chart

plt.plot([0, ks['PercentDataset'].iloc[0]], [0, ks['CumPercentResponse'].iloc[0]], color='blue', marker='o')
plt.plot([0, ks['PercentDataset'].iloc[0]], [0, ks['CumPercentNonResponse'].iloc[0]], color='red', marker='o')

plt.plot(ks['PercentDataset'], ks['CumPercentResponse'], color='blue', label='% Good', marker='o')
plt.plot(ks['PercentDataset'], ks['CumPercentNonResponse'], color='red', label='% Bad', marker='o')
arrow = plt.arrow(ks.loc[ks_val_index, 'PercentDataset'], ks.loc[ks_val_index, 'CumPercentNonResponse'], 0, ks_val, 
          color='green', linewidth=2, label='KS-Statistic Value')

plt.title('KS Chart')

plt.xlabel('% of Dataset')
plt.ylabel('% of Event/Non-Event')

plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 110, 10))

plt.legend(loc='lower right')

plt.grid()

plt.show()

# Distance (KS-statistic) shown in green line below

# tpr = True positive rate
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score


fpr, tpr, thresholds = roc_curve(y_test, y_pred_2)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label = 'Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
print("Roc_auc_score")
print(roc_auc_score(y_test, y_pred_2))
gini_coef = 2*roc_auc_score(y_test, y_pred_2) - 1
print(gini_coef)


# Plot tpr vs 1-fpr
fig, ax = plt.subplots()
plt.plot(roc['tpr'])
plt.plot(roc['1-fpr'], color = 'red')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
ax.set_xticklabels([])

# =============================================================================
# 
# =============================================================================
gain_lift_df = pd.DataFrame(data={'Actual Value': y_test,
                                 'Predicted Value': y_pred_labels,
                                 'Probability':  y_pred_final})
gain_lift_df.head()
gain_lift_df.reset_index(inplace=True)
gain_lift_df.drop(columns='index', axis=1, inplace=True)
# Sort values by probability
gain_lift_df.sort_values(by='Probability', ascending=False, inplace=True)

rows = []

# For loop for computing on deciles
for group in np.array_split(gain_lift_df, 10):
    events = group[group['Actual Value']==1].count()['Actual Value']
    rows.append({'NumCases': len(group), 'NumResponses': events})

lift = pd.DataFrame(rows)

# Cumulative Responses from No. of Responses
lift['CumulativeResponses'] = lift['NumResponses'].cumsum()

# Percent of Events in the decile
lift['PercentEvents'] = round((lift['NumResponses']/lift['NumResponses'].sum())*100, 2)

# Gain calculation from Cumulative Responses
lift['Gain'] = round((lift['CumulativeResponses']/lift['NumResponses'].sum())*100, 2)

# %age Dataset in cumulative format
lift['PercentDataset'] = np.floor((lift['NumCases'].cumsum()/lift['NumCases'].sum())*100)

# Lift calculation from Gain
lift['Lift'] = round((lift['Gain']/(lift['PercentDataset'])), 2)

# Random Lift calculation
lift['Lift(Random)'] = np.ones(len(lift['Lift']))

lift

# Plot Gain chart

plt.plot([0, lift['PercentDataset'].iloc[0]], [0, lift['Gain'].iloc[0]], color='blue', marker='o')
plt.plot([0, lift['PercentDataset'].iloc[0]], [0, lift['PercentDataset'].iloc[0]], color='red', marker='o')

plt.plot(lift['PercentDataset'], lift['Gain'], color='blue', label='% of Cumulative Events (Model)', marker='o')
plt.plot(lift['PercentDataset'], lift['PercentDataset'], color='red', label='% of Cumulative Events (Random)', marker='o')

plt.title('Gain Chart')

plt.xlabel('% of Dataset')
plt.ylabel('Gain')

plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 110, 10))

plt.legend(loc='lower right')

plt.grid()
plt.show()

# Plot Lift curve
plt.plot(lift['PercentDataset'], lift['Lift'], color='blue', label='Lift (Model)', marker='o')
plt.plot(lift['PercentDataset'], lift['Lift(Random)'], color='red', label='Lift (Random)', marker='o')

plt.title('Lift Chart')

plt.xlabel('% of Dataset')
plt.ylabel('Lift')

plt.xticks(np.arange(0, 110, 10))

plt.legend(loc='upper right')

plt.grid()
plt.show()

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred_labels))
print("Precision Score")
print(precision_score(y_test, y_pred_labels))
print("Recall Score")
print(recall_score(y_test, y_pred_labels))
# =============================================================================
# CHeckng fo Lasso tuning again
# =============================================================================

C = [10, 1, .1, .001, .0001, .000001]

for c in C:
    clf = LogisticRegression(penalty='l1', C=c,solver ='liblinear')
    clf.fit(X_train3, y_train)
    print('C:', c)
    print('Coefficient of each feature:', clf.coef_)
    print('Training accuracy:', clf.score(X_train3, y_train))
    print('Test accuracy:', clf.score(X_test3, y_test))
    print('')
# =============================================================================
# End of Logistic Regression, Random Forest starts
# =============================================================================


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 42)

y = app_data_train['TARGET']
X = app_data_train.loc[:, app_data_train.columns != 'TARGET']
X = X.apply(lambda x:round(x,2), axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print ("Train dataset: {0}{1}".format(X_train.shape, y_train.shape))
print ("Test dataset: {0}{1}".format(X_test.shape, y_test.shape))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=False, with_std=False)

scale_cols = X_train.columns

# Fit on Training data and then apply it to every feature set present
X_train[scale_cols] = sc.fit_transform(X_train[scale_cols])
X_test[scale_cols] = sc.transform(X_test[scale_cols])
print ("Train dataset: {0}{1}".format(X_train.shape, y_train.shape))
print ("Test dataset: {0}{1}".format(X_test.shape, y_test.shape))
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:,1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_rf)




feat_imp={'Columns':X_train.columns,'Coefficients':rf.feature_importances_}
feat_imp=pd.DataFrame()
feat_imp['Features']=X_train.columns.values
feat_imp['importance']=rf.feature_importances_
feat_imp=feat_imp.sort_values(by='importance', ascending=False)
sns.set(rc={'figure.figsize':(89.7,55.27)})
sns.barplot(y="Features", x="importance", data=feat_imp)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_rf)
print("Confusion matrix:\n%s" % confusion_matrix)
print("accuracy score",round(accuracy_score(y_test, y_pred_rf),2))
print(classification_report(y_test, y_pred_rf))

#Hyperparameter tuning
import time
rf_tree_cv = RandomForestClassifier(random_state=42)
from sklearn.model_selection import train_test_split, GridSearchCV
np.random.seed(42)
start = time.time()

param_dist = {'max_depth': [2, 3, 4],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'criterion': ['gini', 'entropy']}

cv_rf_tree =GridSearchCV(rf_tree_cv, cv = 10,
                     param_grid=param_dist,
                     n_jobs = 3,scoring='accuracy')

cv_rf_tree.fit(X_train,y_train)
print('Best Parameters using grid search: \n',
     cv_rf_tree.best_params_)
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss,roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import validation_curve
num_est=np.arange(50,1000,100)
train_scoreNum, test_scoreNum = validation_curve(
                                RandomForestClassifier(),
                                X = X_train, y = y_train, 
                                param_name = 'n_estimators', 
                                param_range = num_est, cv = 3)

#tune the model
rf_tree_cv.set_params(criterion = 'gini',max_features = 'auto',max_depth = 2)
rf_tree_cv.fit(X_train,y_train)

y_pred_rf_cv=rf_tree_cv.predict(X_test)
y_pred_rf_cv_pred_proba=rf_tree_cv.predict_proba(X_test)[:,1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_rf_cv)


test_accuracy = accuracy_score(y_test, y_pred_rf_cv2)*100
test_auc_roc = roc_auc_score(y_test, y_pred_rf_cv_pred_proba)*100
test_avg_precision = average_precision_score(y_test, y_pred_rf_cv2)
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred_rf_cv2))
print('Testing accuracy: %.4f %%' % test_accuracy)
print('Testing AUC: %.4f %%' % test_auc_roc)
print('Testing Average Precision Score: %.4f %%' % test_avg_precision)
print(classification_report(y_test, y_pred_rf_cv2))



[fpr, tpr, thr] = roc_curve(y_test, y_pred_rf_cv_pred_proba)

y_pred_rf_cv_train = rf_tree_cv.predict(X_train)
y_pred_proba_rf_cv_train = rf_tree_cv.predict_proba(X_train)[:,1]
train_accuracy = accuracy_score(y_train, y_pred_rf_cv_train)*100
train_auc_roc = roc_auc_score(y_train, y_pred_proba_rf_cv_train)*100
train_avg_precision = average_precision_score(y_train, y_pred_rf_cv_train)
print('Confusion matrix:\n', confusion_matrix(y_train, y_pred_rf_cv_train))
print('Training accuracy: %.4f %%' % train_accuracy)
print('Training AUC: %.4f %%' % train_auc_roc)
print('Training Average Precision Score: %.4f %%' % train_avg_precision)

from numpy import argmax
from numpy import sqrt, mean, std
gmeans = sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thr[ix], gmeans[ix]))
optimal_idx = np.argmin(np.abs(tpr - fpr)) 
optimal_threshold = thr[optimal_idx]
y_pred_rf_cv2 = (y_pred_rf_cv_pred_proba > 0.5 )*1


print("accuracy score",round(accuracy_score(y_test, y_pred_rf_cv2),2))
print(classification_report(y_test, y_pred_rf_cv2))


# number of trees used
print('Number of Trees used : ', model.n_estimators)

from sklearn.metrics import confusion_matrix
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred_rf_cv2))
sns.heatmap(confusion_matrix(y_test, y_pred_rf_cv2), annot=True, cmap='Blues', fmt='g')
plt.ylabel('Actual Label')
_ = plt.xlabel('Predicted Label')

conf_mat = confusion_matrix(y_test, y_pred_rf_cv2)
print("% of False Positive: {}".format(conf_mat[0][1]*100/(conf_mat[0][0] + conf_mat[0][1] + conf_mat[1][0] + conf_mat[1][1])))

print("% of False Negative: {}".format(conf_mat[1][0]*100/(conf_mat[0][0] + conf_mat[0][1] + conf_mat[1][0] + conf_mat[1][1])))

#KS

ks_df = pd.DataFrame(data={'Actual Value': y_test,
                                 'Predicted Value': y_pred_rf_cv2,
                                 'Probability': y_pred_rf_cv_pred_proba})
ks_df.head()
ks_df.reset_index(inplace=True)
ks_df.drop(columns='index', axis=1, inplace=True)

ks_df.sort_values(by='Probability', ascending=False, inplace=True)
rows = []

# For loop for computing on deciles
for group in np.array_split(ks_df, 10):
    events = group[group['Actual Value']==1].count()['Actual Value']
    rows.append({'NumCases': len(group), 'NumResponses': events})

ks = pd.DataFrame(rows)

# No. of Non Responses
ks['NumNonResponses'] = ks['NumCases'] - ks['NumResponses']

# %age of Response
ks['PercentResponse'] = round((ks['NumResponses']/ks['NumResponses'].sum())*100, 2)

# %age of Non Response
ks['PercentNonResponse'] = round((ks['NumNonResponses']/ks['NumNonResponses'].sum())*100, 2)

# Cumulative %age Response
ks['CumPercentResponse'] = round((ks['NumResponses'].cumsum()/ks['NumResponses'].sum())*100, 2)

# Cumulative %age Non Response
ks['CumPercentNonResponse'] = round((ks['NumNonResponses'].cumsum()/ks['NumNonResponses'].sum())*100, 2)

# Difference values: KS = Cumulative % Response - Cumulative % Non Response
ks['KS_Value'] = round(ks['CumPercentResponse'] - ks['CumPercentNonResponse'], 2)

# %age Dataset: decile frequency distribution
ks['PercentDataset'] = round((ks['NumCases'].cumsum()/ks['NumCases'].sum())*100, 2)

# KS-statistic value: max(difference/KS values)
ks_val = ks['KS_Value'].max()

# Index of KS-statistic in column
ks_val_index = ks['KS_Value'].idxmax()

print('KS-Statistic Value: {}'.format(ks_val))

# Plot KS chart

plt.plot([0, ks['PercentDataset'].iloc[0]], [0, ks['CumPercentResponse'].iloc[0]], color='blue', marker='o')
plt.plot([0, ks['PercentDataset'].iloc[0]], [0, ks['CumPercentNonResponse'].iloc[0]], color='red', marker='o')

plt.plot(ks['PercentDataset'], ks['CumPercentResponse'], color='blue', label='% Good', marker='o')
plt.plot(ks['PercentDataset'], ks['CumPercentNonResponse'], color='red', label='% Bad', marker='o')
arrow = plt.arrow(ks.loc[ks_val_index, 'PercentDataset'], ks.loc[ks_val_index, 'CumPercentNonResponse'], 0, ks_val, 
          color='green', linewidth=2, label='KS-Statistic Value')

plt.title('KS Chart')

plt.xlabel('% of Dataset')
plt.ylabel('% of Event/Non-Event')

plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 110, 10))

plt.legend(loc='lower right')

plt.grid()

plt.show()

# Distance (KS-statistic) shown in green line below

# tpr = True positive rate
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score


fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf_cv2)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label = 'Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
print("Roc_auc_score")
print(roc_auc_score(y_test, y_pred_rf_cv2))
gini_coef = 2*roc_auc_score(y_test, y_pred_rf_cv2) - 1
print(gini_coef)

# predict the target on the train dataset
# =============================================================================
# predict_train = model.predict(X_train)
# print('\nTarget on train data',predict_train) 
# 
# # Accuray Score on train dataset
# accuracy_train = accuracy_score(y_train,predict_train)
# print('\naccuracy_score on train dataset : ', accuracy_train)
# 
# # predict the target on the test dataset
# predict_test = model.predict(X_test)
# print('\nTarget on test data',predict_test) 
# 
# # Accuracy Score on test dataset
# accuracy_test = accuracy_score(y_test,predict_test)
# print('\naccuracy_score on test dataset : ', accuracy_test)
# =============================================================================

from sklearn.ensemble import GradientBoostingClassifier
start= time.time()
gbm0 = GradientBoostingClassifier(random_state=10)
gbm0.fit(X_train,y_train)
y_pred_gbm=gbm0.predict(X_test)
end = time.time()
print('Time taken in algo run: {0: .2f}'.format(end - start))

# classification report
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print("accuracy score",round(accuracy_score(y_test, y_pred_gbm),2))
print(classification_report(y_test, y_pred_gbm))

from sklearn.model_selection import train_test_split, GridSearchCV
gbm_cv = GradientBoostingClassifier(learning_rate=0.1,max_features='sqrt', subsample=0.8, random_state=10)
np.random.seed(42)
start = time.time()
param_test1 = {'max_depth':range(1,20,2), 'min_samples_split':range(2,20,3),'min_samples_leaf':range(1,12,2),'n_estimators':range(20,81,10)}
cv_gb_tree1 =GridSearchCV(gbm_cv, cv = 5,
                     param_grid = param_test1, scoring='accuracy',n_jobs=4,iid=False)

cv_gb_tree1.fit(X_train,y_train)
y_pred_gbm_cv1=cv_gb_tree1.predict(X_test)
print('Best Parameters using grid search: \n',
    cv_gb_tree1.best_params_)  
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))

# =============================================================================
# NC Model
# =============================================================================

from sklearn.neighbors import NearestCentroid
modelnc = NearestCentroid()
start= time.time()
modelnc.fit(X_train,y_train)
y_pred_nc=modelnc.predict(X_test)
end = time.time() 
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(modelnc, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

print("accuracy score",round(accuracy_score(y_test, y_pred_nc),2))
print(classification_report(y_test, y_pred_nc))
# =============================================================================
# adaboost  
# =============================================================================
from sklearn.ensemble import AdaBoostClassifier
modelabc = AdaBoostClassifier(random_state=0, learning_rate=1.0, n_estimators=575)
modelabc.fit(X_train, y_train)
y_pred_abc = modelabc.predict(X_test)
y_pred_proba_abc = modelabc.predict_proba(X_test)[:,1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_abc2)

test_accuracy = accuracy_score(y_test, y_pred_abc2)*100
test_auc_roc = roc_auc_score(y_test, y_pred_proba_abc)*100
test_avg_precision = average_precision_score(y_test, y_pred_abc2)
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred_abc2))
print('Testing accuracy: %.4f %%' % test_accuracy)
print('Testing AUC: %.4f %%' % test_auc_roc)
print('Testing Average Precision Score: %.4f %%' % test_avg_precision)
print(classification_report(y_test, y_pred_abc2))



[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba_abc)

y_pred_abc_train = modelabc.predict(X_train)
y_pred_proba_abc_train = modelabc.predict_proba(X_train)[:,1]
train_accuracy = accuracy_score(y_train, y_pred_abc_train)*100
train_auc_roc = roc_auc_score(y_train, y_pred_proba_abc_train)*100
train_avg_precision = average_precision_score(y_train, y_pred_abc_train)
print('Confusion matrix:\n', confusion_matrix(y_train, y_pred_abc_train))
print('Training accuracy: %.4f %%' % train_accuracy)
print('Training AUC: %.4f %%' % train_auc_roc)
print('Training Average Precision Score: %.4f %%' % train_avg_precision)


from numpy import argmax
from numpy import sqrt, mean, std
gmeans = sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thr[ix], gmeans[ix]))
optimal_idx = np.argmin(np.abs(tpr - fpr)) 
optimal_threshold = thr[optimal_idx]
y_pred_abc2 = (y_pred_proba_abc > 0.5 )*1

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(modelabc, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

print("accuracy score",round(accuracy_score(y_test, y_pred_abc2),2))
print(classification_report(y_test, y_pred_abc2))

from sklearn.metrics import confusion_matrix
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred_abc2))
sns.heatmap(confusion_matrix(y_test, y_pred_abc2), annot=True, cmap='Blues', fmt='g')
plt.ylabel('Actual Label')
_ = plt.xlabel('Predicted Label')

conf_mat = confusion_matrix(y_test, y_pred_abc2)
print("% of False Positive: {}".format(conf_mat[0][1]*100/(conf_mat[0][0] + conf_mat[0][1] + conf_mat[1][0] + conf_mat[1][1])))

print("% of False Negative: {}".format(conf_mat[1][0]*100/(conf_mat[0][0] + conf_mat[0][1] + conf_mat[1][0] + conf_mat[1][1])))

#KS

ks_df = pd.DataFrame(data={'Actual Value': y_test,
                                 'Predicted Value': y_pred_abc2,
                                 'Probability': y_pred_proba_abc})
ks_df.head()
ks_df.reset_index(inplace=True)
ks_df.drop(columns='index', axis=1, inplace=True)

ks_df.sort_values(by='Probability', ascending=False, inplace=True)
rows = []

# For loop for computing on deciles
for group in np.array_split(ks_df, 10):
    events = group[group['Actual Value']==1].count()['Actual Value']
    rows.append({'NumCases': len(group), 'NumResponses': events})

ks = pd.DataFrame(rows)

# No. of Non Responses
ks['NumNonResponses'] = ks['NumCases'] - ks['NumResponses']

# %age of Response
ks['PercentResponse'] = round((ks['NumResponses']/ks['NumResponses'].sum())*100, 2)

# %age of Non Response
ks['PercentNonResponse'] = round((ks['NumNonResponses']/ks['NumNonResponses'].sum())*100, 2)

# Cumulative %age Response
ks['CumPercentResponse'] = round((ks['NumResponses'].cumsum()/ks['NumResponses'].sum())*100, 2)

# Cumulative %age Non Response
ks['CumPercentNonResponse'] = round((ks['NumNonResponses'].cumsum()/ks['NumNonResponses'].sum())*100, 2)

# Difference values: KS = Cumulative % Response - Cumulative % Non Response
ks['KS_Value'] = round(ks['CumPercentResponse'] - ks['CumPercentNonResponse'], 2)

# %age Dataset: decile frequency distribution
ks['PercentDataset'] = round((ks['NumCases'].cumsum()/ks['NumCases'].sum())*100, 2)

# KS-statistic value: max(difference/KS values)
ks_val = ks['KS_Value'].max()

# Index of KS-statistic in column
ks_val_index = ks['KS_Value'].idxmax()

print('KS-Statistic Value: {}'.format(ks_val))

# Plot KS chart

plt.plot([0, ks['PercentDataset'].iloc[0]], [0, ks['CumPercentResponse'].iloc[0]], color='blue', marker='o')
plt.plot([0, ks['PercentDataset'].iloc[0]], [0, ks['CumPercentNonResponse'].iloc[0]], color='red', marker='o')

plt.plot(ks['PercentDataset'], ks['CumPercentResponse'], color='blue', label='% Good', marker='o')
plt.plot(ks['PercentDataset'], ks['CumPercentNonResponse'], color='red', label='% Bad', marker='o')
arrow = plt.arrow(ks.loc[ks_val_index, 'PercentDataset'], ks.loc[ks_val_index, 'CumPercentNonResponse'], 0, ks_val, 
          color='green', linewidth=2, label='KS-Statistic Value')

plt.title('KS Chart')

plt.xlabel('% of Dataset')
plt.ylabel('% of Event/Non-Event')

plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 110, 10))

plt.legend(loc='lower right')

plt.grid()

plt.show()

# Distance (KS-statistic) shown in green line below

# tpr = True positive rate
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score


fpr, tpr, thresholds = roc_curve(y_test, y_pred_abc2)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label = 'AdaBoost')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
print("Roc_auc_score")
print(roc_auc_score(y_test, y_pred_abc2))
gini_coef = 2*roc_auc_score(y_test, y_pred_abc2) - 1
print(gini_coef)


# =============================================================================
# etc model
# =============================================================================
from sklearn.ensemble import ExtraTreesClassifier
start= time.time()
etc0 = ExtraTreesClassifier(random_state=10)
etc0.fit(X_train,y_train)
y_pred_etc=etc0.predict(X_test)
end = time.time()
print('Time taken in algo run: {0: .2f}'.format(end - start))
from sklearn.model_selection import RepeatedStratifiedKFold
model = ExtraTreesClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

print("accuracy score",round(accuracy_score(y_test, y_pred_etc),2))
print(classification_report(y_test, y_pred_etc))

# =============================================================================
# 
# =============================================================================
# Modeling
import lightgbm as lgb

# Evaluation of the model
from sklearn.model_selection import KFold

MAX_EVALS = 500
N_FOLDS = 10
# Model with default hyperparameters
model = lgb.LGBMClassifier()
model
from timeit import default_timer as timer

start = timer()
model.fit(X_train,y_train)
train_time = timer() - start

y_pred_lgbm = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_lgbm)

print('The baseline score on the test set is {:.4f}.'.format(auc))
print('The baseline training time is {:.4f} seconds'.format(train_time))
# =============================================================================
# Stacking
# =============================================================================
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from vecstack import stacking
from lightgbm import *

models = [
        
    RandomForestClassifier(criterion = 'gini',max_features = 'auto',max_depth = 2),
        
    LGBMClassifier(colsample_bytree= 0.4453924022919999, min_child_samples= 331, min_child_weight= 0.01, num_leaves= 10, reg_alpha= 7, reg_lambda= 50, subsample= 0.8388825269901063)

]


S_train, S_test = stacking(models, X_train, y_train, X_test,regression=False, mode='oof_pred_bag', 
                           needs_proba=False,save_dir=None, metric=accuracy_score, n_folds=4, stratified=True,
                           shuffle=True,random_state=0,verbose=2)

modelfinal = LogisticRegression()#after validation & tuning

    
modelfinal = modelfinal.fit(S_train, y_train)
y_pred_stack = modelfinal.predict(S_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pred_stack))

y_pred_proba_stack = modelfinal.predict_proba(S_test)[:,1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_stack2)

test_accuracy = accuracy_score(y_test, y_pred_stack2)*100
test_auc_roc = roc_auc_score(y_test, y_pred_proba_stack)*100
test_avg_precision = average_precision_score(y_test, y_pred_stack2)
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred_stack2))
print('Testing accuracy: %.4f %%' % test_accuracy)
print('Testing AUC: %.4f %%' % test_auc_roc)
print('Testing Average Precision Score: %.4f %%' % test_avg_precision)
print(classification_report(y_test, y_pred_stack2))



[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba_abc)

y_pred_abc_train = modelabc.predict(X_train)
y_pred_proba_abc_train = modelabc.predict_proba(X_train)[:,1]
train_accuracy = accuracy_score(y_train, y_pred_abc_train)*100
train_auc_roc = roc_auc_score(y_train, y_pred_proba_abc_train)*100
train_avg_precision = average_precision_score(y_train, y_pred_abc_train)
print('Confusion matrix:\n', confusion_matrix(y_train, y_pred_abc_train))
print('Training accuracy: %.4f %%' % train_accuracy)
print('Training AUC: %.4f %%' % train_auc_roc)
print('Training Average Precision Score: %.4f %%' % train_avg_precision)


from numpy import argmax
from numpy import sqrt, mean, std
gmeans = sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thr[ix], gmeans[ix]))
optimal_idx = np.argmin(np.abs(tpr - fpr)) 
optimal_threshold = thr[optimal_idx]
y_pred_stack2 = (y_pred_proba_stack > 0.5 )*1

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(modelabc, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

print("accuracy score",round(accuracy_score(y_test, y_pred_abc2),2))
print(classification_report(y_test, y_pred_abc2))

from sklearn.metrics import confusion_matrix
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred_abc2))
sns.heatmap(confusion_matrix(y_test, y_pred_abc2), annot=True, cmap='Blues', fmt='g')
plt.ylabel('Actual Label')
_ = plt.xlabel('Predicted Label')

conf_mat = confusion_matrix(y_test, y_pred_abc2)
print("% of False Positive: {}".format(conf_mat[0][1]*100/(conf_mat[0][0] + conf_mat[0][1] + conf_mat[1][0] + conf_mat[1][1])))

print("% of False Negative: {}".format(conf_mat[1][0]*100/(conf_mat[0][0] + conf_mat[0][1] + conf_mat[1][0] + conf_mat[1][1])))

#KS

ks_df = pd.DataFrame(data={'Actual Value': y_test,
                                 'Predicted Value': y_pred_abc2,
                                 'Probability': y_pred_proba_abc})
ks_df.head()
ks_df.reset_index(inplace=True)
ks_df.drop(columns='index', axis=1, inplace=True)

ks_df.sort_values(by='Probability', ascending=False, inplace=True)
rows = []

# For loop for computing on deciles
for group in np.array_split(ks_df, 10):
    events = group[group['Actual Value']==1].count()['Actual Value']
    rows.append({'NumCases': len(group), 'NumResponses': events})

ks = pd.DataFrame(rows)

# No. of Non Responses
ks['NumNonResponses'] = ks['NumCases'] - ks['NumResponses']

# %age of Response
ks['PercentResponse'] = round((ks['NumResponses']/ks['NumResponses'].sum())*100, 2)

# %age of Non Response
ks['PercentNonResponse'] = round((ks['NumNonResponses']/ks['NumNonResponses'].sum())*100, 2)

# Cumulative %age Response
ks['CumPercentResponse'] = round((ks['NumResponses'].cumsum()/ks['NumResponses'].sum())*100, 2)

# Cumulative %age Non Response
ks['CumPercentNonResponse'] = round((ks['NumNonResponses'].cumsum()/ks['NumNonResponses'].sum())*100, 2)

# Difference values: KS = Cumulative % Response - Cumulative % Non Response
ks['KS_Value'] = round(ks['CumPercentResponse'] - ks['CumPercentNonResponse'], 2)

# %age Dataset: decile frequency distribution
ks['PercentDataset'] = round((ks['NumCases'].cumsum()/ks['NumCases'].sum())*100, 2)

# KS-statistic value: max(difference/KS values)
ks_val = ks['KS_Value'].max()

# Index of KS-statistic in column
ks_val_index = ks['KS_Value'].idxmax()

print('KS-Statistic Value: {}'.format(ks_val))

# Plot KS chart

plt.plot([0, ks['PercentDataset'].iloc[0]], [0, ks['CumPercentResponse'].iloc[0]], color='blue', marker='o')
plt.plot([0, ks['PercentDataset'].iloc[0]], [0, ks['CumPercentNonResponse'].iloc[0]], color='red', marker='o')

plt.plot(ks['PercentDataset'], ks['CumPercentResponse'], color='blue', label='% Good', marker='o')
plt.plot(ks['PercentDataset'], ks['CumPercentNonResponse'], color='red', label='% Bad', marker='o')
arrow = plt.arrow(ks.loc[ks_val_index, 'PercentDataset'], ks.loc[ks_val_index, 'CumPercentNonResponse'], 0, ks_val, 
          color='green', linewidth=2, label='KS-Statistic Value')

plt.title('KS Chart')

plt.xlabel('% of Dataset')
plt.ylabel('% of Event/Non-Event')

plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 110, 10))

plt.legend(loc='lower right')

plt.grid()

plt.show()

# Distance (KS-statistic) shown in green line below

# tpr = True positive rate
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score


fpr, tpr, thresholds = roc_curve(y_test, y_pred_abc2)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label = 'AdaBoost')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
print("Roc_auc_score")
print(roc_auc_score(y_test, y_pred_abc2))
gini_coef = 2*roc_auc_score(y_test, y_pred_abc2) - 1
print(gini_coef)

# =============================================================================
# 
# =============================================================================
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from mlxtend.classifier import StackingCVClassifier
xgb = XGBClassifier()
lgbm = LGBMClassifier()
rf = RandomForestClassifier()
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
ridge = Ridge()
lasso = Lasso()
svr = SVR(kernel='linear')
stack = StackingCVClassifier(classifiers=(ridge, lasso, svr, rf, lgbm, xgb),
                            meta_classifier=xgb, cv=12,
                            use_features_in_secondary=True,
                            store_train_meta_features=True,
                            shuffle=False,
                            random_state=42)

stack.fit(X_train, y_train)

X_test.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12']
pred = stack.predict(X_test)
score = r2_score(y_test, pred)

# =============================================================================
# 
# =============================================================================
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot
 
# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('lr', LogisticRegression()))
	level0.append(('knn', KNeighborsClassifier()))
	level0.append(('cart', DecisionTreeClassifier()))
	level0.append(('svm', SVC()))
	level0.append(('bayes', GaussianNB()))
	# define meta learner model
	level1 = LogisticRegression()
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model
 
# get a list of models to evaluate
def get_models():
	models = dict()
	models['lr'] = LogisticRegression()
	models['knn'] = KNeighborsClassifier()
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	models['stacking'] = get_stacking()
	return models
 
# evaluate a give model using cross-validation
def evaluate_model(model):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores
 

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()





from sklearn.datasets import make_classification
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# define the base models
level0 = list()
level0.append(('lrreg', LogisticRegression()))
level0.append(('knn', KNeighborsClassifier()))
level0.append(('cart', DecisionTreeClassifier()))
level0.append(('svm', SVC()))
level0.append(('bayes', GaussianNB()))
# define meta learner model
level1 = LogisticRegression()
# define the stacking ensemble
modelst = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
# fit the model on all available data
modelst.fit(X_train, y_train)
# make a prediction for one example
data = [[2.47475454,0.40165523,1.68081787,2.88940715,0.91704519,-3.07950644,4.39961206,0.72464273,-4.86563631,-6.06338084,-1.22209949,-0.4699618,1.01222748,-0.6899355,-0.53000581,6.86966784,-3.27211075,-6.59044146,-2.21290585,-3.139579]]
yhat = modelst.predict(data)
print('Predicted Class: %d' % (yhat))


# =============================================================================
# Score CHAMPION MODEL
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
y_pred_champ = modelfinal.predict()


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# =============================================================================
# 
# =============================================================================
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
# =============================================================================
X = app_data_train
y = app_data_train['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
# =============================================================================
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
models

# =============================================================================
# 
# =============================================================================
from pycaret.classification import *
clf1 = setup(data = app_data_train, target = 'TARGET')
# tuning LightGBM Model

ridge = create_model('ridge')
lda = create_model('lda')
gbc = create_model('gbc')
xgboost = create_model('xgboost')
# stacking models
stacker = stack_models(estimator_list = [ridge,lda,gbc], meta_model = xgboost)

lr = LogisticRegression(class_weight = {0:1,1:11.3},C=0.001,penalty='l1', solver ='liblinear')#after validation & tuning
lr.fit(X_train, y_train)
# Computing prediction on 'X_test' test dataset, outputs predicted labels
y_stck = lr.predict(aa)
y_pred_proba = lr.predict_proba(X_test)[:,1]

B = app_data_test[final_features]
B = B.drop(columns = cols_to_remove)
B = B.reset_index()

y_stack = lr.predict(B)
yo = pd.DataFrame(data = y_stack)
yo = yo.reset_index()

yoo = B.merge(yo, on='index', how='inner')
yoo.to_csv(r'C:\Users\jain vibhanshu\Desktop\VJ\Caselets\1 - Python Ensemble\Problem\Problem\Data\final_pred2.csv')

# =============================================================================
# lgbm
# =============================================================================
fit_params={"early_stopping_rounds":30, 
            "eval_metric" : 'auc', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100,
            'categorical_feature': 'auto'}
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

n_HP_points_to_test = 100

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test, 
    n_iter=n_HP_points_to_test,
    scoring='roc_auc',
    cv=3,
    refit=True,
    random_state=314,
    verbose=True)

gs.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

opt_parameters = {'colsample_bytree': 0.4453924022919999, 'min_child_samples': 331, 'min_child_weight': 0.01, 'num_leaves': 10, 'reg_alpha': 7, 'reg_lambda': 50, 'subsample': 0.8388825269901063}
clf_sw = lgb.LGBMClassifier(**clf.get_params())
#set optimal parameters
clf_sw.set_params(**opt_parameters)
gs_sample_weight = GridSearchCV(estimator=clf_sw, 
                                param_grid={'scale_pos_weight':[1,2,6,12]},
                                scoring='roc_auc',
                                cv=5,
                                refit=True,
                                verbose=True)
gs_sample_weight.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs_sample_weight.best_score_, gs_sample_weight.best_params_))

print("Valid+-Std     Train  :   Parameters")
for i in np.argsort(gs_sample_weight.cv_results_['mean_test_score'])[-5:]:
    print('{1:.3f}+-{3:.3f}     {2:.3f}   :  {0}'.format(gs_sample_weight.cv_results_['params'][i], 
                                    gs_sample_weight.cv_results_['mean_test_score'][i], 
                                    gs_sample_weight.cv_results_['mean_train_score'][i],
                                    gs_sample_weight.cv_results_['std_test_score'][i]))
    
#Configure from the HP optimisation
#clf_final = lgb.LGBMClassifier(**gs.best_estimator_.get_params())
def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3
#Configure locally from hardcoded values
clf_final = lgb.LGBMClassifier(**clf.get_params())
#set optimal parameters
clf_final.set_params(**opt_parameters)

#Train the final model with learning rate decay
clf_final.fit(X_train, y_train, **fit_params, callbacks=[lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_0995)])

probabilities = clf_final.predict_proba(app_data_test.drop(['SK_ID_CURR'], axis=1))
submission = pd.DataFrame({
    'SK_ID_CURR': app_data_test['SK_ID_CURR'],
    'TARGET':     [ row[1] for row in probabilities]
})

y_pred_gbm = clf_final.predict(X_test)
y_pred_proba_gbm = clf_final.predict_proba(X_test)[:,1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_gbm2)

test_accuracy = accuracy_score(y_test, y_pred_gbm2)*100
test_auc_roc = roc_auc_score(y_test, y_pred_proba_gbm)*100
test_avg_precision = average_precision_score(y_test, y_pred_gbm2)
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred_gbm2))
print('Testing accuracy: %.4f %%' % test_accuracy)
print('Testing AUC: %.4f %%' % test_auc_roc)
print('Testing Average Precision Score: %.4f %%' % test_avg_precision)
print(classification_report(y_test, y_pred_gbm2))



[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba_abc)

y_pred_gbm_train = clf_final.predict(X_train)
y_pred_proba_gbm_train = clf_final.predict_proba(X_train)[:,1]
train_accuracy = accuracy_score(y_train, y_pred_gbm_train)*100
train_auc_roc = roc_auc_score(y_train, y_pred_proba_gbm_train)*100
train_avg_precision = average_precision_score(y_train, y_pred_gbm_train)
print('Confusion matrix:\n', confusion_matrix(y_train, y_pred_gbm_train))
print('Training accuracy: %.4f %%' % train_accuracy)
print('Training AUC: %.4f %%' % train_auc_roc)
print('Training Average Precision Score: %.4f %%' % train_avg_precision)


from numpy import argmax
from numpy import sqrt, mean, std
gmeans = sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thr[ix], gmeans[ix]))
optimal_idx = np.argmin(np.abs(tpr - fpr)) 
optimal_threshold = thr[optimal_idx]
y_pred_gbm2 = (y_pred_proba_gbm > 0.5 )*1

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(modelabc, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

print("accuracy score",round(accuracy_score(y_test, y_pred_abc2),2))
print(classification_report(y_test, y_pred_abc2))

from sklearn.metrics import confusion_matrix
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred_gbm2))
sns.heatmap(confusion_matrix(y_test, y_pred_gbm2), annot=True, cmap='Blues', fmt='g')
plt.ylabel('Actual Label')
_ = plt.xlabel('Predicted Label')

conf_mat = confusion_matrix(y_test, y_pred_gbm2)
print("% of False Positive: {}".format(conf_mat[0][1]*100/(conf_mat[0][0] + conf_mat[0][1] + conf_mat[1][0] + conf_mat[1][1])))

print("% of False Negative: {}".format(conf_mat[1][0]*100/(conf_mat[0][0] + conf_mat[0][1] + conf_mat[1][0] + conf_mat[1][1])))

#KS

ks_df = pd.DataFrame(data={'Actual Value': y_test,
                                 'Predicted Value': y_pred_gbm2,
                                 'Probability': y_pred_proba_gbm})
ks_df.head()
ks_df.reset_index(inplace=True)
ks_df.drop(columns='index', axis=1, inplace=True)

ks_df.sort_values(by='Probability', ascending=False, inplace=True)
rows = []

# For loop for computing on deciles
for group in np.array_split(ks_df, 10):
    events = group[group['Actual Value']==1].count()['Actual Value']
    rows.append({'NumCases': len(group), 'NumResponses': events})

ks = pd.DataFrame(rows)

# No. of Non Responses
ks['NumNonResponses'] = ks['NumCases'] - ks['NumResponses']

# %age of Response
ks['PercentResponse'] = round((ks['NumResponses']/ks['NumResponses'].sum())*100, 2)

# %age of Non Response
ks['PercentNonResponse'] = round((ks['NumNonResponses']/ks['NumNonResponses'].sum())*100, 2)

# Cumulative %age Response
ks['CumPercentResponse'] = round((ks['NumResponses'].cumsum()/ks['NumResponses'].sum())*100, 2)

# Cumulative %age Non Response
ks['CumPercentNonResponse'] = round((ks['NumNonResponses'].cumsum()/ks['NumNonResponses'].sum())*100, 2)

# Difference values: KS = Cumulative % Response - Cumulative % Non Response
ks['KS_Value'] = round(ks['CumPercentResponse'] - ks['CumPercentNonResponse'], 2)

# %age Dataset: decile frequency distribution
ks['PercentDataset'] = round((ks['NumCases'].cumsum()/ks['NumCases'].sum())*100, 2)

# KS-statistic value: max(difference/KS values)
ks_val = ks['KS_Value'].max()

# Index of KS-statistic in column
ks_val_index = ks['KS_Value'].idxmax()

print('KS-Statistic Value: {}'.format(ks_val))

# Plot KS chart

plt.plot([0, ks['PercentDataset'].iloc[0]], [0, ks['CumPercentResponse'].iloc[0]], color='blue', marker='o')
plt.plot([0, ks['PercentDataset'].iloc[0]], [0, ks['CumPercentNonResponse'].iloc[0]], color='red', marker='o')

plt.plot(ks['PercentDataset'], ks['CumPercentResponse'], color='blue', label='% Good', marker='o')
plt.plot(ks['PercentDataset'], ks['CumPercentNonResponse'], color='red', label='% Bad', marker='o')
arrow = plt.arrow(ks.loc[ks_val_index, 'PercentDataset'], ks.loc[ks_val_index, 'CumPercentNonResponse'], 0, ks_val, 
          color='green', linewidth=2, label='KS-Statistic Value')

plt.title('KS Chart')

plt.xlabel('% of Dataset')
plt.ylabel('% of Event/Non-Event')

plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 110, 10))

plt.legend(loc='lower right')

plt.grid()

plt.show()

# Distance (KS-statistic) shown in green line below

# tpr = True positive rate
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score


fpr, tpr, thresholds = roc_curve(y_test, y_pred_gbm2)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label = 'AdaBoost')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
print("Roc_auc_score")
print(roc_auc_score(y_test, y_pred_gbm2))
gini_coef = 2*roc_auc_score(y_test, y_pred_gbm2) - 1
print(gini_coef)

