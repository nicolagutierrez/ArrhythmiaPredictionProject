#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nicola Gutierrez
"""

import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as skm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix, confusion_matrix

import time
from datetime import timedelta

warnings.filterwarnings('ignore')
sns.set_style("white")

#Set True in order to show plots, tables and data details
show_data_analysis = False
#%%
"""########################################################################################################"""

def simple_plot(values, x_label, y_label):
    values.plot(color = "black")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def countplot_bars(data, title, i):
    plt.figure(figsize=(7, 8))
    ax = sns.countplot(x=data[i], palette="rocket_r")
    ax.set_title(title, pad=18, fontsize=18)
    sns.despine(right=True)
    plt.show()

def countplot_bars_series(data, title):
    plt.figure(figsize=(7, 8))
    ax = sns.countplot(x=data, palette="flare")
    ax.set_title(title, pad=18, fontsize=18)
    
    total = float(len(data))
    for p in ax.patches:
       height = p.get_height()
       ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.1f}%'.format((height/total)*100), ha="center")
    sns.despine(right=True)
    plt.show()

def plot_outliers(df):
    sns.set(rc={'figure.figsize':(11.7,5.27)})
    sns.boxplot(data=df[["QRS_Dur","P-R_Int","Q-T_Int","T_Int","P_Int"]])
    plt.show()
    """P-R interval: Average duration between onset of P and Q waves in msec., linear
       Maybe outliers but keep them"""
    
    sns.set(rc={'figure.figsize':(11.7,5.27)})
    sns.boxplot(data=df[["AVR190","AVR191","AVR192","AVR193","AVR194","AVR195","AVR196","AVR197","AVR198","AVR199"]])
    plt.show()
    
    sns.set(rc={'figure.figsize':(11.7,5.27)})
    sns.boxplot(data=df[["AVL200","AVL201","AVL202","AVL203","AVL204","AVL205","AVL206","AVL207","AVL208","AVL209"]])
    plt.show()
    
    sns.set(rc={'figure.figsize':(11.7,5.27)})
    sns.boxplot(data=df[["AVF210","AVF211","AVF212","AVF213","AVF214","AVF215","AVF216","AVF217","AVF218","AVF219"]])
    plt.show()
    
    pair_grid(final_df, ['Age', 'Sex', 'Height', 'Weight'], 'Sex')
    plt.show()
    
    print('\nHEIGHT values', sorted(final_df['Height'], reverse=True)[:10]) #not plausible values
    print('\nWEIGHT values', sorted(final_df['Weight'], reverse=True)[:10], '\n') #plausible values

def pair_grid(df, _vars, _hue):
    g = sns.PairGrid(df, vars=_vars, hue=_hue, palette="rocket_r")
    g.map(plt.scatter, alpha=0.8)
    g.add_legend()
    plt.show()
    
    
def plot_conf_matrix(y_test, y_pred):
    """Multiclass data will be treated as if binarized under a one-vs-rest transformation. 
        Returned confusion matrices will be in the order of sorted unique labels in the union of (y_true, y_pred).
        true negatives  -> 0,0   false positives -> 0,1
        false negatives -> 1,0   true positives  -> 1,1"""    
    conf_mat_all = multilabel_confusion_matrix(y_test, y_pred)
    y_tmp = sorted(np.array(y_test.drop_duplicates()))
    i = 0
    
    for conf_mat in conf_mat_all:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(np.eye(2), annot=conf_mat, fmt='g', annot_kws={'size': 35},
                    cmap=sns.color_palette(['#df9fbf', '#993366'], as_cmap=True), cbar=False, ax=ax)

        ax.tick_params(labelsize=15, length=0)
        
        title = 'Confusion Matrix: ' + str(y_tmp[i])
        i = i + 1
        ax.set_title(title, size=24, pad=20)
        
        ax.set_xlabel('Test Values', size=18)
        ax.set_ylabel('Predicted Values', size=18)
        
        additional_texts = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
        for text_elt, additional_text in zip(ax.texts, additional_texts):
            ax.text(*text_elt.get_position(), '\n' + additional_text, 
                    color=text_elt.get_color(), ha='center', va='top', size=10)

def check_duplicated_rows(df):
    if df.duplicated().sum():
        print("\n!!!!!The dataset contains duplicated rows: ", df[df.duplicated()],  "!!!!!")

def check_nan_values(df):
    nv = pd.isnull(df).sum().sum()
    if nv:
        print("\n!!!!! The dataset contains NaN values: ", nv, "!!!!!")

def imputation(df):
    # Replace Missing values 
    # Imputing the remaining missing values with median of the values in the column
    imputer_mean = SimpleImputer(missing_values=np.NaN, strategy='median')
    imputer = imputer_mean.fit(df)
    df_tmp = imputer.transform(df)
    df_tmp = pd.DataFrame(df_tmp)
    return df_tmp

"""########################################################################################################"""
#%%
"""--------------------------------------- #0 - Pre-Processing ----------------------------------------"""
starting_time = time.time() #Start data pre-processing & EDA

"""The path refers to the workspace ArrhythmiaPredictionProject/"""
df = pd.read_csv("./data/arrhythmia.data",header=None)

print("\n--------- DATAFRAME'S INFO ---------")
print(df.info()) #10, 11, 12, 13, 14 have 'object' type
print("------------------------------------\n")

"""One of the 'object' classes (13) will not be considered (~80% are missing values)
   The other 4 classes are numerical"""

# Split the frame as Data attribute and Class
df_data = df.iloc[:,:-1]
df_class = df.iloc[:,-1] # target -> 'class'

#Replace ? by "Other" (16th classes)
#df_class = df_class.replace('?', 16) #Not necessary, there are no missing values in the target class

# Replace ? by NaN
df_data = df_data.replace('?', np.NaN)

#Check NaN values distribution
if show_data_analysis:
    simple_plot(pd.isnull(df_data).sum(), 'Columns', 'Total number of null value in each column')
    nv = pd.isnull(df_data).sum().sort_values(ascending=False)
    simple_plot(pd.isnull(df_data).sum()[7:17], 'Columns', 'Total number of null value in each column') #ZOOM

"""
Column 13 contains 376 missing values out of total 452 instances. 
So we will drop column 13. other attributes have comparatively less null values. 
So instead of droping, we will replace the null value of other attributes with their mean values.
"""

# Remove unwanted columns
# Deleting the attributes having more than 40% missing values.
thresh =  (len(df_data) * 0.4)
df_data.dropna(thresh = thresh, axis = 1, inplace = True)

#Imputation
df_data = imputation(df_data)

#Dataframe final version (data + target)
final_df = pd.concat([df_data, df_class], axis=1)

check_duplicated_rows(final_df)
check_nan_values(final_df)

#Columns names
columns_names=["Age","Sex","Height","Weight","QRS_Dur",
"P-R_Int","Q-T_Int","T_Int","P_Int","QRS","T","P","J","Heart_Rate",
"Q_Wave","R_Wave","S_Wave","R'_Wave","S'_Wave","Int_Def","Rag_R_Nom",
"Diph_R_Nom","Rag_P_Nom","Diph_P_Nom","Rag_T_Nom","Diph_T_Nom", 
"DII00", "DII01","DII02", "DII03", "DII04","DII05","DII06","DII07","DII08","DII09","DII10","DII11",
"DIII00","DIII01","DIII02", "DIII03", "DIII04","DIII05","DIII06","DIII07","DIII08","DIII09","DIII10","DIII11",
"AVR00","AVR01","AVR02","AVR03","AVR04","AVR05","AVR06","AVR07","AVR08","AVR09","AVR10","AVR11",
"AVL00","AVL01","AVL02","AVL03","AVL04","AVL05","AVL06","AVL07","AVL08","AVL09","AVL10","AVL11",
"AVF00","AVF01","AVF02","AVF03","AVF04","AVF05","AVF06","AVF07","AVF08","AVF09","AVF10","AVF11",
"V100","V101","V102","V103","V104","V105","V106","V107","V108","V109","V110","V111",
"V200","V201","V202","V203","V204","V205","V206","V207","V208","V209","V210","V211",
"V300","V301","V302","V303","V304","V305","V306","V307","V308","V309","V310","V311",
"V400","V401","V402","V403","V404","V405","V406","V407","V408","V409","V410","V411",
"V500","V501","V502","V503","V504","V505","V506","V507","V508","V509","V510","V511",
"V600","V601","V602","V603","V604","V605","V606","V607","V608","V609","V610","V611",
"JJ_Wave","Amp_Q_Wave","Amp_R_Wave","Amp_S_Wave","R_Prime_Wave","S_Prime_Wave","P_Wave","T_Wave",
"QRSA","QRSTA","DII170","DII171","DII172","DII173","DII174","DII175","DII176","DII177","DII178","DII179",
"DIII180","DIII181","DIII182","DIII183","DIII184","DIII185","DIII186","DIII187","DIII188","DIII189",
"AVR190","AVR191","AVR192","AVR193","AVR194","AVR195","AVR196","AVR197","AVR198","AVR199",
"AVL200","AVL201","AVL202","AVL203","AVL204","AVL205","AVL206","AVL207","AVL208","AVL209",
"AVF210","AVF211","AVF212","AVF213","AVF214","AVF215","AVF216","AVF217","AVF218","AVF219",
"V1220","V1221","V1222","V1223","V1224","V1225","V1226","V1227","V1228","V1229",
"V2230","V2231","V2232","V2233","V2234","V2235","V2236","V2237","V2238","V2239",
"V3240","V3241","V3242","V3243","V3244","V3245","V3246","V3247","V3248","V3249",
"V4250","V4251","V4252","V4253","V4254","V4255","V4256","V4257","V4258","V4259",
"V5260","V5261","V5262","V5263","V5264","V5265","V5266","V5267","V5268","V5269",
"V6270","V6271","V6272","V6273","V6274","V6275","V6276","V6277","V6278","V6279",
"class"]

#Adding Column names to dataset
final_df.columns = columns_names

"""----------------------------------------------------------------------------------------------------"""
#%%
"""--------------------------------------------- #1 - EDA ---------------------------------------------"""
"""Analyzing dataset to summarize their main characteristics.
    Making List of all the type of Arrythmia corresponsing to their class label"""

#List with class names
class_names = ["Normal", 
               "Ischemic changes (CAD)", 
               "Old Anterior Myocardial Infraction",
               "Old Inferior Myocardial Infraction",
               "Sinus tachycardy", 
               "Sinus bradycardy", 
               "Ventricular Premature Contraction (PVC)",
               "Supraventricular Premature Contraction",
               "Left Boundle branch block",
               "Right boundle branch block",
               "1.Degree AtrioVentricular block",   #NotPresent [11]
               "2.Degree AV block",                 #NotPresent [12]
               "3.Degree AV block",                 #NotPresent [13]
               "Left Ventricule hypertrophy",
               "Atrial Fibrillation or Flutter",
               "Other"]
    
"""Analyzing the dataset and check how many examples we have for each class:
   we need to sort our dataset with respect to class attributes to count the number of 
   instances available for each class"""

if show_data_analysis:
    #Counting the number of instances for each class
    class_distribution_binary = final_df.copy()
    class_distribution_binary["class"] = class_distribution_binary["class"].astype('str')
    class_distribution_binary["class"] = np.where(class_distribution_binary["class"] == "1", "normal", "arrythmia")
    countplot_bars(class_distribution_binary, 'Target distribution', 'class')

    class_distribution = final_df.copy()
    #In order to remove normal class
    class_distribution = class_distribution.drop(class_distribution[class_distribution["class"] == 1].index)
    countplot_bars(class_distribution,'Arrhythmia classes distribution', 'class')

    """Heavily biased towards the normal cases (245/452), it will be perform over sampling to balance the classes.
       It will be only perform on training data to avoid information leakage.
       There are also 12 different types of arrhythmias and 3 other type of arrthmias are not present in this dataset. [11, 12, 13]"""

    #Looking for pairwise relationships and outliers
    plot_outliers(final_df)

#Mean of world height : ~170cm
final_df['Height']=final_df['Height'].replace(608,170)
final_df['Height']=final_df['Height'].replace(780,170)

if show_data_analysis:
    pair_grid(final_df, ['Age', 'Sex', 'Height', 'Weight'], 'Sex') #Re-scaled
    
"""There are some strong correlation (>0.9) either negative or positive
   There is NO significant orrelation (<0.6) with class (target)"""
corr_matrix = final_df.corr()
#corr_matrix.to_csv("./extra/correlation_matrix/correlation_matrix.csv")

correlation_list = list(dict())
for c in corr_matrix.columns:
    for r in corr_matrix.index:
        if (c == r):
            break
        if (abs(corr_matrix[c][r]) > 0.9):
            correlation_list.append({"col" : c, "row" : r, "correlation" : corr_matrix[c][r]})

reduced_df = final_df.copy()
for record in correlation_list:
    if record["col"] in reduced_df.columns:
        reduced_df.drop(columns = record["col"], inplace = True)

print("\n----- FINAL DATAFRAME'S INFO -----")
print(final_df.info())
print("----------------------------------\n")

print("\n----- REDUCED DATAFRAME'S INFO -----")
print(reduced_df.info())
print("------------------------------------\n")

# #Reduce the normal cases contribution doesn't improve performances
# reduced_df = reduced_df.sort_values(by=['class'])
# for i in range(24):
#     reduced_df.drop(reduced_df.index[[0]], inplace=True)
# reduced_df = shuffle(reduced_df)

#Split dataset in train set and test set
y = reduced_df["class"]
X = reduced_df.drop(columns=['class']).values
# y = final_df["class"]
# X = final_df.drop(columns=['class']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

ending_time = time.time()
print("\n---------- Elapsed time for EDA & pre-processing:",timedelta(seconds=ending_time - starting_time),"----------\n")

"""----------------------------------------------------------------------------------------------------"""
#%%
"""--------------------------------------- #2 - CV : GridSearch ---------------------------------------"""
models = [KNeighborsClassifier(weights='distance'),
          DecisionTreeClassifier(class_weight='balanced', random_state=42),
          LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced', random_state=42),
          SVC(class_weight='balanced', probability=True, random_state=42)]
          
models_names = ['K-Nearest Neighbors',
                'Decision Tree',
                'Softmax Regression',
                'Support Vector Machine']

models_hparams = [{'n_neighbors': list(range(1, 10, 2))},
                  
                  {'max_depth' : [3, 4, 5, 7, 10], 'criterion': ['gini', 'entropy']},
                  
                  {'penalty': ['l1', 'l2'], 'C': [1e-5, 5e-5, 1e-4, 5e-4, 1.0]},
                  
                  {'C': [1e-4, 1e-2, 1.0, 1e1, 1e2], 'gamma': ['scale', 1e-4, 1e-3, 1e-2], 'kernel': ['linear', 'rbf']}]                

chosen_hparams = list()
estimators = list()

for model, model_name, hparams in zip(models, models_names, models_hparams):
    
        print("\n************ ", model_name, " ************")
        starting_time = time.time()
        clf = GridSearchCV(estimator=model, param_grid=hparams, scoring='accuracy', cv=7)
        clf.fit(X_train, y_train)
        ending_time = time.time()
        chosen_hparams.append(clf.best_params_)
        estimators.append((model_name, clf.best_score_, clf.best_estimator_))
        
        for hparam in hparams:
            print('\n---> best value for hyperparameter', hparam, ': ', clf.best_params_.get(hparam))
        print('\n---------- Elapsed time for GridSearch: ', timedelta(seconds=ending_time - starting_time), "\n\n")

final_estimators = list()

#First part of Performance evaluation [w/o Stacking Classifier]
for model_est in estimators:
    model_name = model_est[0]
    model = model_est[2]
    scores = cross_validate(model, X_train, y_train, cv=7, scoring=('f1_weighted', 'accuracy'))
    print('---> The cross-validated Accuracy of', model_name, '  is: ', np.mean(scores['test_accuracy']))
    print('---> The cross-validated F1-Score of', model_name, '  is: ', np.mean(scores['test_f1_weighted']),'\n')
    final_estimators.append((model_name, np.mean(scores['test_f1_weighted']), np.mean(scores['test_accuracy']), model))

"""----------------------------------------------------------------------------------------------------"""
#%%
"""------------------------------------------- #3 - Ensemble ------------------------------------------"""
#Ensemble -> StackingClassifier (w/3 weak-learners)
final_estimators.sort(key=lambda i:i[2],reverse=True)

print('The Stacking Classifier is made with :')

best_clfs = list()
for clf in final_estimators[0:3]:
    best_clfs.append((clf[0], clf[3]))
    print(clf[0])
    
clf_stack = StackingClassifier(estimators = best_clfs, final_estimator = LogisticRegression())

"""----------------------------------------------------------------------------------------------------"""
#%%
"""--------------------------------- #4 - CV : Performance Evaluation ---------------------------------"""
print("\n************ Stacking Model ************")

scores = cross_validate(clf_stack, X_train, y_train, cv=7, scoring=('f1_weighted', 'accuracy'))
print('---> Cross-validated Accuracy of Stacking Model is ', np.mean(scores['test_accuracy']),)
print('---> Cross-validated F1-Score of Stacking Model is ', np.mean(scores['test_f1_weighted']), '\n')

final_estimators.append(('Stacking Classifier', np.mean(scores['test_f1_weighted']), np.mean(scores['test_accuracy']), clf_stack))

final_estimators.sort(key=lambda i:i[1],reverse=True)
print('\n-----> The best 3 model are :')
for model_est in final_estimators[0:3]:
    print(model_est[0])

"""----------------------------------------------------------------------------------------------------"""
#%%
"""----------------------------------------- #5 - Final Model -----------------------------------------"""
final_model_name = final_estimators[0][0]
final_model = final_estimators[0][3]
print('\n\n-----> The Final Model selected is:', final_model_name, '<-----')

"""----------------------------------------------------------------------------------------------------"""
#%%
"""---------------------------------------- #6 - Final Training ---------------------------------------"""
final_model.fit(X_train, y_train)

"""----------------------------------------------------------------------------------------------------"""
#%%
"""-------------------------------- #7 - Test [Pre-processing : scaling] ------------------------------"""
X_test = scaler.transform(X_test)

"""----------------------------------------------------------------------------------------------------"""
#%%
"""-------------------------------------------- #8 - Testing ------------------------------------------"""
y_pred = final_model.predict(X_test)

print('\n----------------------------> FINAL TESTING <----------------------------')
print('---> Accuracy is ', accuracy_score(y_test, y_pred))
print('---> Precision is ', precision_score(y_test, y_pred, average='weighted'))
print('---> Recall is ', recall_score(y_test, y_pred, average='weighted'))
print('---> F1-Score is ', f1_score(y_test, y_pred, average='weighted'))

"""----------------------------------------------------------------------------------------------------"""
#%%    
"""--------------------------------------- #9 - Confusion Matrix --------------------------------------"""
if show_data_analysis:
    countplot_bars_series(y_pred,'y prediction')
    countplot_bars_series(y_test,'y test')
    cm = confusion_matrix(y_test, y_pred)
    plot_conf_matrix(y_test, y_pred)

print('\n\n', skm.classification_report(y_test,y_pred), '\n\n')
#%%
