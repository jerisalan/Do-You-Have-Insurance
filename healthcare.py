import os
import sys
import json
import math
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def missing_values_table(df): 
    """Function to display the percentage and number of missing values in a Data Frame."""
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    print(mis_val_table_ren_columns)

def print_stats(expected, predicted, learning_algo):
    """Function to display the classification results and metrics."""
    print('Confusion matrix for: %s'%learning_algo)
    print(metrics.confusion_matrix(expected, predicted))
    print('Actual Recall')
    print(metrics.recall_score(expected, predicted, average='macro'))
    print('Actual Accuracy Score')
    print(metrics.accuracy_score(expected, predicted))
    print('Actual Precision Score')
    print(metrics.precision_score(expected, predicted, average='macro'))
    print('Actual F1 Score')
    print(metrics.f1_score(expected, predicted, average='macro'))    

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """Function to plot the confusion matrix."""
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def preprocess_data(data, test_data_ind):
    """Function to preprocess the dataset."""
    # Continuous variables are assigned the mean as the missing data value
    data['intp'].fillna((data['intp'].mean()), inplace=True)
    data['pincp'].fillna((data['pincp'].mean()), inplace=True)
    data['povpip'].fillna((data['povpip'].mean()), inplace=True)
    data['retp'].fillna((data['retp'].mean()), inplace=True)
    data['wkhp'].fillna((data['wkhp'].mean()), inplace=True)
    data['pap'].fillna((data['pap'].mean()), inplace=True)

    # Categorical variables are assigned the mode for missing data value
    data = data.fillna({"esr": data["esr"].mode()[0]})
    data = data.fillna({"schl": data["schl"].mode()[0]})    

    # Certain income parameter values were found to be negative, so they have been set to their column means
    data.loc[data['intp'] < 0, 'intp'] = data['intp'].mean()
    data.loc[data['pincp'] < 0, 'pincp'] = data['pincp'].mean()
    
    # Apply label encoding to the categorical variables
    data = label_encode_dataset(data)

    if(test_data_ind != 1): # Label encoding should be done only for training dataset. If test_data_ind is 1, then this column will not be present
        hicov_le = preprocessing.LabelEncoder()
        data['hicov'] = hicov_le.fit_transform(data['hicov'])
        
    # Factorize the continuous data columns into bins
    data = factorize_data(data)

    return data

def label_encode_dataset(data):
    """This function label encodes the categorical columns."""
    cit_le = preprocessing.LabelEncoder()
    dear_deye_le = preprocessing.LabelEncoder()
    esr_le = preprocessing.LabelEncoder()
    mar_le = preprocessing.LabelEncoder()
    race_le = preprocessing.LabelEncoder()
    schl_le = preprocessing.LabelEncoder()
    sex_le = preprocessing.LabelEncoder()
    st_le = preprocessing.LabelEncoder()
    vet_le = preprocessing.LabelEncoder()
    puma_le = preprocessing.LabelEncoder()

    data['cit'] = cit_le.fit_transform(data['cit'])
    data['dear'] = dear_deye_le.fit_transform(data['dear'])
    data['deye'] = dear_deye_le.fit_transform(data['deye'])
    data['esr'] = esr_le.fit_transform(data['esr'])
    data['mar'] = mar_le.fit_transform(data['mar'])
    data['puma'] = puma_le.fit_transform(data['puma'])
    data['race'] = race_le.fit_transform(data['race'])
    data['schl'] = schl_le.fit_transform(data['schl'])
    data['sex'] = sex_le.fit_transform(data['sex'])
    data['st'] = st_le.fit_transform(data['st'])
    data['vet'] = vet_le.fit_transform(data['vet'])

    return data

def drop_columns(data, columns_list):
    """Delete a list of columns from the DataFrame."""
    return data.drop(columns_list, axis=1)

def factorize_data(data):
    """This function is used to bin continuous variables into respective bins."""
    data['agep'] = (data['agep'] / 10).astype(int) 
    data['pincp'] = (data['pincp'] / 50000).astype(int) 
    data['intp'] = (data['intp'] / 10000).astype(int) 
    data['pap'] = (data['pap'] / 1000).astype(int) 
    data['retp'] = (data['retp'] / 4000).astype(int) 
    data['wkhp'] = (data['wkhp'] / 20).astype(int) 
    data['povpip'] = (data['povpip'] / 100).astype(int) 

    return data

# Read training and testing dataset
data = pd.read_json('train.json', orient = 'records')
test_data = pd.read_json('test.json', orient='records')

# Exploratory data analysis
print(data.describe())
missing_values_table(data)

print(test_data.describe())
missing_values_table(test_data)

print(data['cit'].unique())
print(data['dear'].unique())
print(data['deye'].unique())
print(data['esr'].unique())
print(data['hicov'].unique())
print(data['mar'].unique())
print(data['race'].unique())
print(data['schl'].unique())
print(data['sex'].unique())
print(data['st'].unique())
print(data['vet'].unique())

print(pd.crosstab(data['race'],data['hicov']).apply(lambda r: r/r.sum(), axis=1))
print(pd.crosstab(data['esr'],data['hicov']).apply(lambda r: r/r.sum(), axis=1))

print(data.groupby('esr').mean())
print(data.groupby('mar').mean())
print(data.groupby('hicov').mean())
print(data.groupby('dear').describe())
print(data.groupby('deye').describe())

# Visualizations and Plotting

a = data[(data['agep']>=20) & (data['agep']<30) & (data['sex'] == 'Male')]['retp']
b = data[(data['agep']>=30) & (data['agep']<40) & (data['sex'] == 'Male')]['retp']
c = data[(data['agep']>=40) & (data['agep']<100) & (data['sex'] == 'Male')]['retp']
sns.distplot(a,bins=100,  hist=False,  label="20 to 30 (Age in years)"  );
sns.distplot(b,bins=100,  hist=False,  label="30 to 40"  );
g = sns.distplot(c,bins=100,  hist=False,  label="40 to 100" );
sns.despine()
g.tick_params(labelsize=12,labelcolor="black")
plt.xlim(0, 200000)
plt.xticks([50000,100000,150000,200000], ['$50K', '$100K', '$120K', '$150K','$200K'])
plt.title("Income Distribution\nby Age, Males", fontname='Ubuntu', fontsize=14,
            fontstyle='italic', fontweight='bold')
plt.xlabel('Retirement Income')
plt.legend()
plt.show()

a = data[(data['agep']>=20) & (data['agep']<30) & (data['sex'] == 'Female')]['retp']
b = data[(data['agep']>=30) & (data['agep']<40) & (data['sex'] == 'Female')]['retp']
c = data[(data['agep']>=40) & (data['agep']<100) & (data['sex'] == 'Female')]['retp']
sns.distplot(a,bins=100,  hist=False,  label="20 to 30 (Age in years)"  );
sns.distplot(b,bins=100,  hist=False,  label="30 to 40"  );
g = sns.distplot(c,bins=100,  hist=False,  label="40 to 100" );
sns.despine()
g.tick_params(labelsize=12,labelcolor="black")
plt.xlim(0, 200000)
plt.xticks([50000,100000,150000,200000], ['$50K', '$100K', '$120K', '$150K','$200K'])
plt.title("Income Distribution\nby Age, Females", fontname='Ubuntu', fontsize=14,
            fontstyle='italic', fontweight='bold')
plt.xlabel('Retirement Income')
plt.legend()
plt.show()

# f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.set(color_codes=True)
sns.set_palette(sns.color_palette("muted"))
sns.distplot(data["pincp"].dropna(), color="orange");
plt.xlabel('Personal Income')
plt.show()
sns.distplot(data['retp'].dropna(), color="olive")
plt.xlabel('Retirement Income')
plt.show()
sns.distplot(data['intp'].dropna(), color="gold")
plt.xlabel('Interest Income')
plt.show()
sns.distplot(data['pap'].dropna(), color="teal")
plt.xlabel('Public Assistance Income')
plt.show()

sns.countplot(x='hicov', data=data, palette='hls')
plt.show()
sns.countplot(x='esr', data=data, palette='hls')
plt.show()
sns.countplot(x='mar', data=data, palette='hls')
plt.show()
sns.countplot(x='race', data=data, palette='hls')
plt.show()
sns.countplot(x='schl', data=data, palette='hls')
plt.show()

pd.crosstab(data['mar'],data['hicov']).plot(kind='bar')
plt.xlabel('Marriage Status')
plt.ylabel('Frequency with or without insurance')
plt.show()

# Pre-process the dataset
data = preprocess_data(data, 0)
test_data = preprocess_data(test_data, 1)

test_ids = test_data['id'] 
test_data = drop_columns(test_data, ['id'])

# Split the dataset into training and validation set. Check Analysis.docx for the split ratio decision
ins = data[data.hicov == 0] # People with healthcare
no_ins = data[data.hicov == 1] # People without healthcare

# There are 366684 rows in the dataset with people having healthcare
ins_train = ins.sample(10000) # randomly sample 10k entries
ins_val = ins[~ins.index.isin(ins_train.index)] # This will contain the remaining

# There are 24598 rows in the dataset with people having no healthcare
noins_train = no_ins.head(6000)
noins_val = no_ins.tail(18598)


train_df = pd.concat([ins_train, noins_train], axis=0)
train_target_df = train_df['hicov']
train_df = train_df.drop(['id', 'hicov'], axis = 1)
print('Training data shape - ', train_df.shape)

val_df = pd.concat([ins_val, noins_val], axis=0)
val_target_df = val_df['hicov']
val_df = val_df.drop(['id', 'hicov'], axis = 1)
print('Validation data shape - ', val_df.shape)

print(pd.DataFrame(train_df).dtypes)
print(pd.DataFrame(train_df).head())

# Do not uncomment this section. This will take a long time to execute as hyperparameter optimization is being done here.
# Use Grid Serach CV for Decision Tree Hyperparameter Tuning
# param_grid = {"max_depth": [3, 5, 7, 9],
#               "max_features": ["sqrt"],
#               "min_samples_leaf": np.arange(3,20,3),
#               "criterion": ["gini", "entropy"]}

# dtree = tree.DecisionTreeClassifier(random_state=42)
# dtree_cv = GridSearchCV(dtree, param_grid, cv=10)
# dtree_cv.fit(val_df, val_target_df)

# print(dtree_cv.best_params_)
# print(dtree_cv.best_score_)

# max_depth = 9
# max_features = "sqrt"
# min_samples_leaf = 12
# criterion = "entropy"

# Use Grid Serach CV for K Nearest Neighbors Hyperparameter Tuning
# param_grid = {"n_neighbors": [5, 7, 9],
#               "algorithm": ["ball_tree", "kd_tree"],              
#               "leaf_size": [10, 20, 30]}

# knn = KNeighborsClassifier()
# knn_cv = GridSearchCV(knn, param_grid, cv=5)
# knn_cv.fit(val_df, val_target_df)

# print(knn_cv.best_params_)
# print(knn_cv.best_score_)

# n_neighbors = 9
# algorithm = "ball_tree"
# leaf_size = 10

# Use Grid Serach CV for Random Forest Hyperparameter Tuning
# param_grid = {  "n_estimators"      : [10, 20, 30],
#                 "criterion"         : ["gini", "entropy"],
#                 "max_features"      : ['sqrt', 'log2'],
#                 "max_depth"         : [3, 6, 9] }

# rf = RandomForestClassifier()
# rf_cv = GridSearchCV(rf, param_grid, cv=5)
# rf_cv.fit(val_df, val_target_df)

# print(rf_cv.best_params_)
# print(rf_cv.best_score_)

# exit()
# n_neighbors = 9
# algorithm = "ball_tree"
# leaf_size = 10

# End of hyperparameter optimization section

dt = tree.DecisionTreeClassifier(random_state=17, max_depth=9, max_features="sqrt", criterion="entropy", min_samples_leaf=12)
knn = KNeighborsClassifier(n_neighbors=9, algorithm="ball_tree", leaf_size=10)
mlp = MLPClassifier(hidden_layer_sizes=100, activation="relu", solver="adam", alpha=0.001, learning_rate_init=0.001, max_iter=500)
rf = RandomForestClassifier(criterion="entropy", n_estimators=30, max_depth=9, max_features="log2")

clf = rf.fit(train_df, train_target_df)
predictions = clf.predict(val_df)

# print_stats(val_target_df, predictions, 'Decision Tree')
# print_stats(val_target_df, predictions, 'K Nearest Neighbor')
# print_stats(val_target_df, predictions, 'Neural Network')
print_stats(val_target_df, predictions, 'Random Forest')

plt.figure()
plot_confusion_matrix(metrics.confusion_matrix(val_target_df, predictions), classes=["With Healthcare", "Without Healthcare"], title='Confusion matrix')
plt.show()

# Predict the results for the given test dataset
result = []
pred = clf.predict(test_data)
pred = pd.Series(pred)

# Concatenate the dataframes and then write back as a JSON file
output = pd.concat([test_ids, pred], axis=1)
output.columns = ['id', 'pred']

out = output.to_json(orient='records')
with open('results.json', 'w') as f:
    f.write(out)

