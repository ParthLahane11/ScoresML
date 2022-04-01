import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from Preprocessing import Preprocess_several as pre
from sklearn.metrics import mean_squared_error
#Load and shuffle datasets
Dataset_mat = pd.read_csv('../Datasets/student-mat.csv', delimiter = ';')
Dataset_por = pd.read_csv('../Datasets/student-por.csv', delimiter = ';')
Dataset_whole = pd.concat([Dataset_mat, Dataset_por])
Dataset_whole = Dataset_whole.sample(frac= 1).reset_index(drop = True)

#Separate Test set and Covalidation set
Dataset_train, Dataset_test = train_test_split(Dataset_whole, test_size=0.2)
Dataset_train, Dataset_cvd = train_test_split(Dataset_train, test_size=0.25)

#Seprate Labels from training data
Dataset_train_Features, Dataset_train_Labels = pre.separate_labels(Dataset_train)
Dataset_test_Features, Dataset_test_Labels = pre.separate_labels((Dataset_test))
Dataset_cvd_Features, Dataset_cvd_Labels = pre.separate_labels(Dataset_cvd)

#Preprocess the data and see correlations
Dataset_train_processed = pre.preprocess_pipeline(dataset = Dataset_train_Features)
Dataset_test_processed = pre.preprocess_pipeline(dataset = Dataset_test_Features)
Dataset_cvd_processed = pre.preprocess_pipeline(dataset = Dataset_cvd_Features)

#Remove unnecessary Features
corr_matrix = Dataset_train_processed.corr()
corr_dict = corr_matrix[57].sort_values(ascending=False).to_dict()
Dataset_train_int = pre.remove_uninteresting_labels(Dataset_train_processed, corr_dict)
Dataset_test_int = pre.remove_uninteresting_labels(Dataset_test_processed, corr_dict)
Dataset_cvd_int = pre.remove_uninteresting_labels(Dataset_cvd_processed, corr_dict)

#Train on Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(Dataset_train_processed, Dataset_train_Labels)
predictions = np.round_(lin_reg.predict(Dataset_cvd_processed))
print(lin_reg.score(Dataset_train_processed, Dataset_train_Labels))
#Plot Train and Test data
