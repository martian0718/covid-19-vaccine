import sys
assert sys.version_info >= (3, 5)
import time
import warnings
import os

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn.linear_model

import seaborn as sns

import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

from scipy.stats import pearsonr
warnings.filterwarnings("ignore")

#Options and parameters.
dataFileName = "country_vaccinations.csv"

#Functions used
def clearNullsWIthMean(dataFrame, featureName):
	dataFrame[featureName].fillna(dataFrame[featureName].mean(), inplace = True)
	dataFrame.fillna(0, inplace = True)
	return None

def showDescriptiveStats(dataFrame, featureName):
	print(featureName, " | set stats: ")
	print(dataFrame[featureName].describe())
	print('\n')
	return None

#Read the data file
df = pd.read_csv(dataFileName)

#Split the data set into a train and test set.
train_df, test_df = train_test_split(df,shuffle = True, test_size = 0.95, random_state=17)

#Display descriptive statistics of the data set.
features = list(train_df.columns)
for feat in features:
	print(feat)
	showDescriptiveStats(train_df, feat) #TODO: show all of the features.

#Display boxplots
fig, axs = plt.subplots(1, len(features))
xAxis = 0
for dataset in features:
    axs[xAxis].boxplot(test_df[dataset], notch = True)
    axs[xAxis].set_title(dataset)
    xAxis += 1
fig.subplots_adjust(left=0, right=2, bottom=0, top=1, hspace=0.5, wspace=0.5) #adjust fig for increased spacing
plt.show() #Show the boxplots

#Display pairplots
#TODO: ^

#Display scatterplot
train_df, test_df = train_test_split(dataFileName,shuffle=True,test_size=0.95, random_state=17)
#Display correlation among variables
corr, _ = pearsonr(X, y)
print('Pearsons correlation: %.3f' % corr)
