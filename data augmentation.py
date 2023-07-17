import pandas as pd
from pandas_ods_reader import read_ods

import numpy

from copy import copy, deepcopy

import random
import os

sheet_index = 1
dataframe = read_ods("data.ods" , sheet_index )


#shuffle
dataframe = dataframe.sample(frac=1).reset_index(drop=True)

print("shape", dataframe.shape)


def attribute_influence(df, col, thresh):
    w_greater = 0
    w_less = 0

    greater = 0
    less = 0
    for i in range(len(df)):
        if df.iloc[i, col] > thresh:
            greater += 1
            if df.iloc[i, 2] == "Wedge":
                w_greater +=1
        else:
            less += 1
            if df.iloc[i, 2] == "Wedge":
                w_less+=1

    if greater == 0 or less == 0: return 0
    return abs(w_greater/greater - w_less/less)


def find_topk_attributes_v1(df, k):
    res = []

    
    for i in range(3, df.shape[1]):
        thresh = df.iloc[:,i].mean()
        res.append((i, attribute_influence(df, i, thresh)))
    
    res = sorted(res, key = lambda x: x[1], reverse = True)
    temp = list(map(lambda x: (list(df.columns.values)[x[0]], x[1]),res))
    for i in range(len(temp)):
        print(str(i + 1), temp[i][0], round(temp[i][1],3), sep = ",")
    print([res[i][0] for i in range(k)])
    return [res[i][0] for i in range(k)]


def find_topk_attributes_v2(df, k): #best treshold search
    res = []

    
    for i in range(3, df.shape[1]):
        thresh = df.iloc[:,i].min()
        start = thresh
        end = df.iloc[:,i].max()
        div = 10
        step = (end - start)/div

        best_value = 0
        best_thresh = None
        for iter in range(5):
            for j in range(div + 1):
                value = attribute_influence(df, i, thresh)
                if value > best_value:
                    best_value = value
                    best_thresh = thresh
                thresh += step
            
            start = max(start, best_thresh - step)   
            end = min(end, best_thresh + step)
            step = (end - start)/div

            print("best thresh", best_thresh, "best value", best_value)
        res.append((i, best_value))
    
    res = sorted(res, key = lambda x: x[1], reverse = True)
    temp = list(map(lambda x: (list(df.columns.values)[x[0]], x[1]),res))
    for i in range(len(temp)):
        print(str(i + 1), temp[i][0], round(temp[i][1],3), sep = ",")
    print([res[i][0] for i in range(k)])
    return [res[i][0] for i in range(k)]


def find_topk_attributes_chi2(df, k, n = 2, log = False): #with chi squared
    res = []
    
    expected = 438/n #wedges in dataset = 438
    
    for feature in range(3, df.shape[1]):
        max_value = df.iloc[:, feature].max()
        min_value = df.iloc[:, feature].min()
        step = (max_value - min_value)/n

        distribution = [0] * n #number of wedges per partition
        
        for i in range(len(df)):
            for d in range(n):
                if min_value + d*step <= df.iloc[i, feature] <= min_value + (d + 1)*step:
                    if df.iloc[i, 2] == "Wedge":
                        distribution[d] += 1
                    break

        #chi squared
        chi2 = 0
        for value in distribution:
            chi2 += ((value - expected)**2)/expected

        res.append( (feature, chi2) )
    res = sorted(res, key = lambda x: x[1], reverse = True)

    #print
    if log:
        temp = list(map(lambda x: (list(df.columns.values)[x[0]], x[1]),res))
        for i in range(len(temp)):
            print(str(i + 1), temp[i][0], round(temp[i][1],3), sep = ",")
        print("n"+str(n)+":",[res[i][0] for i in range(k)])

    return [res[i][0] for i in range(k)]    


def row_distance(df, row1, row2, attributes):
    if df.iloc[row1, 2] != df.iloc[row2, 2]:
        return 9999999999

    row1_trim = df.iloc[row1, attributes]
    row2_trim = df.iloc[row2, attributes]

    return numpy.linalg.norm(row1_trim - row2_trim)


def merge_rows(df, row1, row2, attributes):
    new_row = [None] * df.shape[1]
    new_row[0] = -1
    v = row2 - row1
    for i in range(1, df.shape[1]):
        b = not (i in attributes)
        new_row[i] = df.iloc[row1 + b*v, i]


    a = df.append(pd.DataFrame([new_row], columns=df.columns.values),ignore_index=True)

    for i in range(1, df.shape[1]):
        b = i in attributes
        new_row[i] = df.iloc[row1 + b*v, i]
    
    b = a.append(pd.DataFrame([new_row], columns=a.columns.values),ignore_index=True)

    return b


def data_augmentation(df, attributes, val_set, filename, skip_augmentation = False):
    new_dataframe = copy(df)

    if not skip_augmentation:
        for i in range(df.shape[0] - 1):
            min_dist = 9999999999
            closest_row = None
            for j in range(i + 1, df.shape[0]):
                dist = row_distance(df, i, j, attributes)
                if  dist < min_dist:
                    min_dist = dist
                    closest_row = j
        
            if closest_row:
                new_dataframe = merge_rows(new_dataframe, i, closest_row, attributes)

            print(str((i+1)/len(df) * 100) + "%    ", end = "\r")

    
    new_dataframe = new_dataframe.append(val_set)
    print("done                     ")
    try:
        with pd.ExcelWriter(filename, engine="odf") as doc:
            new_dataframe.to_excel(doc, sheet_name="Sheet1", index = False)
    except:
        input("close excel, then press enter.")
        with pd.ExcelWriter(filename, engine="odf") as doc:
            new_dataframe.to_excel(doc, sheet_name="Sheet1", index = False)




random.seed(47)

K = 5
TOP_ATTRIBUTES = {
    "random" : random.choices(list(range(3, 51)), k = 48),
    "average threshold" :  [37, 7, 39, 5, 6, 25, 26, 13, 23, 3, 38, 4, 44, 21, 22, 11, 24, 20, 43, 36, 12, 9, 40, 19, 33, 48, 49, 18, 32, 41, 35, 50, 45, 15, 27, 14, 28, 34, 47, 30, 42, 29, 8, 46, 17, 10, 31, 16], 
    "maximizer threshold": [37, 5, 7, 39, 6, 36, 3, 4, 38, 25, 21, 26, 23, 24, 20, 22, 44, 43, 12, 13, 17, 18, 27, 28, 29, 32, 33, 34, 35, 40, 46, 48, 49, 50, 11, 41, 8, 9, 10, 14, 15, 16, 19, 30, 31, 42, 45, 47],
    "SHAP analysis" :     [37, 9, 40, 4, 3, 48, 8, 23, 17, 32, 25, 18, 29, 5, 11, 38, 13, 50, 16, 10, 26, 6, 21, 7, 24, 46, 28, 19, 39, 43, 36, 14, 15, 22, 34, 42, 33, 45, 44, 31, 20, 47, 30, 49, 41, 35, 27, 12],
    "chi2 n2" :            [3, 4, 5, 6, 7, 39, 44, 43, 37, 20, 36, 24, 11, 21, 25, 26, 22, 23, 13, 12, 41, 38, 9, 42, 15, 18, 27, 49, 45, 10, 46, 48, 17, 16, 31, 34, 35, 30, 47, 40, 50, 14, 19, 32, 28, 29, 33, 8],
    "chi2 n4" :            [44, 6, 7, 39, 5, 3, 4, 43, 36, 37, 9, 20, 25, 22, 12, 24, 13, 11, 23, 21, 26, 38, 41, 10, 49, 15, 27, 16, 42, 8, 45, 18, 19, 14, 17, 46, 34, 28, 33, 31, 35, 48, 40, 30, 32, 29, 50, 47],     
    "chi2 n8" :            [44, 6, 7, 39, 4, 3, 5, 43, 9, 36, 37, 20, 25, 12, 22, 21, 13, 24, 11, 23, 26, 38, 27, 41, 15, 49, 10, 34, 14, 16, 8, 45, 42, 33, 28, 19, 18, 17, 32, 46, 31, 35, 48, 40, 29, 30, 50, 47],
    "chi2 n16" :           [44, 6, 7, 39, 4, 3, 5, 43, 9, 37, 36, 24, 20, 25, 13, 22, 11, 12, 21, 23, 26, 27, 38, 41, 34, 49, 10, 14, 8, 15, 16, 42, 45, 32, 33, 28, 18, 19, 17, 46, 31, 35, 48, 29, 40, 30, 50, 47],
    "chi2 n32" :           [44, 6, 7, 39, 4, 3, 5, 43, 9, 37, 36, 12, 24, 20, 22, 13, 25, 11, 27, 21, 23, 26, 38, 34, 41, 8, 49, 10, 14, 15, 42, 16, 32, 45, 33, 28, 19, 18, 17, 46, 31, 35, 29, 48, 30, 40, 50, 47],
    "chi2 n64" :           [44, 6, 7, 39, 4, 3, 5, 9, 43, 37, 36, 12, 24, 20, 22, 25, 27, 13, 11, 21, 23, 26, 38, 34, 41, 8, 49, 10, 15, 14, 42, 16, 32, 45, 33, 28, 18, 19, 46, 17, 31, 35, 29, 30, 48, 40, 50, 47]
}

N_ATTRIBUTES = [12, 24]
METHODS = ["chi2 b2", "chi2 n4", "SHAP analysis"]


#no augmentation
for k in range(K):
    dfcopy = copy(dataframe)
    size = len(dataframe)//K
    dfcopy = dfcopy.drop(dfcopy.index[k*size : (k+1)*size])

    data_augmentation(dfcopy, None, dataframe.iloc[k*size : (k+1)*size], "datasets/k({}).ods".format(k+1) , skip_augmentation= True)


#augmented normalize
cols_to_norm = list(dataframe.columns.values)[3:]
dataframe[cols_to_norm] = (dataframe[cols_to_norm]-dataframe[cols_to_norm].mean())/dataframe[cols_to_norm].std()

for method in METHODS:
    for n_attributes in N_ATTRIBUTES:
        for k in range(K):
            dfcopy = copy(dataframe)
            size = len(dataframe)//K
            dfcopy = dfcopy.drop(dfcopy.index[k*size : (k+1)*size])

            data_augmentation(dfcopy, TOP_ATTRIBUTES[method][:n_attributes], dataframe.iloc[k*size : (k+1)*size], "datasets/augmented m({}) a({}) k({}) N.ods".format(method, n_attributes, k + 1) )           
