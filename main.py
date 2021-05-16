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

#Establish the helper functions
def clearNullsWithMean(dataFrame, featureName):
    dataFrame[featureName].fillna(dataFrame[featureName].mean(), inplace = True)
    dataFrame.fillna(0, inplace = True)
    return None

def clearNullsWithZero(dataFrame, featureName):
    dataFrame[featureName].fillna(0, inplace = True)
    dataFrame.fillna(0, inplace = True)
    return None

#===========================================================================================================
#Read the data file
df = pd.read_csv(dataFileName)
df.head()

#Clean the data set
#TODO ^
#Clean null values
#Clean non-2021 dates

#Split the data set into a train and test set.
train_df, test_df = train_test_split(df,shuffle = True, test_size = 0.95, random_state=17)

#===========================================================================================================#===========================================================================================================
#Display descriptive statistics of the data set.
features = list(train_df.columns)
featuresWithData = [
	'total_vaccinations',
	'people_vaccinated',
	'people_fully_vaccinated',
	'daily_vaccinations_raw',
	'daily_vaccinations',
	'total_vaccinations_per_hundred',
	'people_vaccinated_per_hundred',
	'people_fully_vaccinated_per_hundred',
	'daily_vaccinations_per_million'
]
for feat in featuresWithData:
	print(feat)
	showDescriptiveStats(train_df, feat)

#===========================================================================================================
#Display boxplots
fig, axs = plt.subplots(1, len(featuresWithData))
xAxis = 0
for dataset in featuresWithData:
    axs[xAxis].boxplot(test_df[dataset], notch = True)
    axs[xAxis].set_title(dataset)
    xAxis += 1
fig.subplots_adjust(left=0, right=2, bottom=0, top=1, hspace=0.5, wspace=0.5) #adjust fig for increased spacing
plt.show() #Show the boxplots

#===========================================================================================================
#Display pairplots

sns.set()
sns.pairplot(test_df[featuresWithData]) #show pairplots

#===========================================================================================================
#Display scatterplot
df.shape
df.isna().sum()

#Convert dates into the date type and then into the integer type.
x = train_df['date']
x = [pd.to_datetime(d).value for d in x]

#Clean the set of nulls/NaNs
clearNullsWithZero(df, 'date')
clearNullsWithZero(df, 'people_vaccinated')

#Reshape the Y set.
y = train_df['people_vaccinated']
x2 = np.array(x).reshape(-1,1)
#print(x2)

#Load the linear regression model
model = sklearn.linear_model.LinearRegression()
model.fit(x2,y)

#display the R-Squared score
r_sq = model.score(x2, y)
print("R-Squared: ", r_sq)

plt.scatter(x2, y, color='black', label='observed')
plt.plot(x2, model.predict(x2), label='fit', color='Green', linewidth=2)
plt.xlabel('Date')
plt.ylabel('People Vaccinated')
plt.title('Regression')
plt.legend(loc='best')
plt.show()
#===========================================================================================================
#Display correlation among variables
corr, _ = pearsonr(X, y)
print('Pearsons correlation: %.3f' % corr)
