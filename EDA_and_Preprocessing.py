#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: klaus
"""

import pandas as pd
import numpy as np
import scipy as sp
import math as mt
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split

"""############################################################################################"""

def check_duplicated_rows(df):
    if df.duplicated().sum():
        print("\n!!!!!The dataset contains duplicated rows: " + str(df[df.duplicated()]) + "!!!!!")
    else:
        print("\n-----The dataset doesn't contain duplicated rows-----")

def check_nan_values(df):
    nv = pd.isnull(df).sum().sum()
    if nv:
        print("\n!!!!!The dataset contains NaN values: " + str(nv) + "!!!!!")
    else:
        print("\n-----The dataset doesn't containes NaN values-----")

def simple_plot(values, x_label, y_label):
    values.plot()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

"""############################################################################################"""


#df = pd.read_csv("./data/arrhythmia.data",header=None)


#Check duplicate rows
check_duplicated_rows(df)

#Replacing ? with np.nan value
df = df.replace('?', np.NaN)
check_nan_values(df)

#Check NaN values distribution
#simple_plot(pd.isnull(df).sum(), 'Columns', 'Total number of null value in each column')
#nv = pd.isnull(df).sum().sort_values(ascending=False)
#simple_plot(pd.isnull(df).sum()[7:17], 'Columns', 'Total number of null value in each column') #ZOOM

"""
Column 13 contains 376 missing values out of total 452 instances. 
So we will drop column 13. other attributes have comparatively less null values. 
So instead of droping, we will replace the null value of other attributes with their mean values.
"""

#Dropping the column 13
df.drop(columns = 13, inplace=True)

""" IMPUTATION ---------------------------------------------------------------------------------"""
#Make copy to avoid changing original data (when Imputing)
df_complete = df.copy()

#Make new columns indicating what will be imputed
cols_with_nan = (col for col in df_complete.columns if df_complete[col].isnull().any())
for col in cols_with_nan:
     df_complete[col] = df_complete[col].isnull()

#Imputation
imputer = SimpleImputer() #default -> SimpleImputer(missing_values=np.nan, strategy='mean')
df_complete = pd.DataFrame(imputer.fit_transform(df_complete))
#df_complete.columns = df.columns

check_nan_values(df_complete)

""" Generating final dataset -------------------------------------------------------------------"""
#Creating column names -> class is the target
final_df_columns=["Age","Sex","Height","Weight","QRS_Dur",
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
"V6270","V6271","V6272","V6273","V6274","V6275","V6276","V6277","V6278","V6279","class"]

#Adding Column names to dataset
df_complete.columns=final_df_columns
df_complete.to_csv("./data/arrhythmia_final.data")
head_df_complete = df_complete.head()


""" EDA ----------------------------------------------------------------------------------------"""
"""Analyzing data sets to summarize their main characteristics.
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
               "1.Degree AtrioVentricular block",
               "2.Degree AV block",
               "3.Degree AV block",
               "Left Ventricule hypertrophy",
               "Atrial Fibrillation or Flutter",
               "Not Considerated"]
    
"""Analyzing the dataset and check how many examples we have for each class:
    we need to sort our dataset with respect to class attributes to count the number of 
    instances available for each class"""
   
#Counting the number of instances for each class
sns.countplot(x ='class', data = df_complete)
plt.show()

class_distribution = df_complete.copy()
class_distribution["class"] = class_distribution["class"].astype('str')
class_distribution["class"] = np.where(class_distribution["class"] == "1.0", "normal", "arrhythmia")
sns.countplot(x ='class', data = class_distribution, palette='Pastel1')
plt.show()

"""We found that Of the 452 examples, 245 refers to "normal" people. 
    We also have 12 different types of arrhythmias and 3 other type of arrthmias are not 
    present in our dataset. [11, 12, 13]"""

# """Handling Outliers & Data Visualization"""
# #looking for pairwise relationships and outliers
# g = sns.PairGrid(df_complete, vars=['Age', 'Sex', 'Height', 'Weight'],hue='Sex', palette='Pastel1')
# g.map(plt.scatter, alpha=0.8)
# g.add_legend();
# #According to scatter plots, there are few outliers in 'height' and 'weight' attributes.

# maxHeightValues = sorted(df_complete['Height'], reverse=True)[:10] #implausible values
# df_complete['Height']=df_complete['Height'].replace(608,108)
# df_complete['Height']=df_complete['Height'].replace(780,180)

# maxWeightValues = sorted(df_complete['Weight'], reverse=True)[:10] #plausible values


# """SPLIT THE DATASET---------------------------------------------------------------------------"""
# y = df_complete["class"]
# X = df_complete.drop(columns=['class']).values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8, stratify=y)
















    