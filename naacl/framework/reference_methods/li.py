import sklearn.linear_model
import sklearn.neighbors
import sklearn.ensemble
import sklearn.svm
import scipy.stats as st
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import sys
import itertools
from naacl.framework import util
import pandas as pd


'''
Main approach advocated in 

	@article{li_inferring_2017,
		title = {Inferring {Affective} {Meanings} of {Words} from {Word} {Embedding}},
		volume = {PP},
		doi = {10.1109/TAFFC.2017.2723012},
		number = {99},
		journal = {IEEE Transactions on Affective Computing},
		author = {Li, Minglei and Lu, Qin and Long, Yunfei and Gui, Lin},
		year = {2017},
		pages = {1--1},
		}

which is mainly ridge regression on top of the embedding vectors with the 
scikit-learn default parameters.
'''

class Multi_Target_Regressor():

	def __init__(self, init_fun=sklearn.linear_model.Ridge):
		'''
		init_fun			A scikit-learn instantiation of a model such as
							linear regression or ridge regression.
		'''
		self.model=init_fun()
		self.var_names=None

	def fit(self, features, labels):
		self.model.fit(features, labels)
		self.var_names=labels.columns

	def predict(self, features):
		preds=pd.DataFrame(self.model.predict(features), index=features.index,
			columns=self.var_names)
		return preds

	def eval(self, train_features, train_labels, test_features, test_labels):
		'''
		Assumes pandas data frames as input
		'''
		self.model.fit(train_features, train_labels)
		preds=pd.DataFrame(data=self.model.predict(test_features), 
				index=test_features.index, columns=list(test_labels))
		performance=pd.Series(index=list(test_labels)+['Average'])
		for var in list(test_labels):
			performance.loc[var]=st.pearsonr(preds.loc[:,var], test_labels.loc[:,var])[0]
		performance.loc['Average']=np.mean(performance[:-1])
		return performance

	def crossvalidate(self, features, labels, k_folds):	
		k=0
		results_df=pd.DataFrame(columns=labels.columns)
		for fold in util.k_folds_split(features, labels, k=k_folds):
			k+=1
			print(k)
			results_df.loc[k]=self.eval(*fold)
		results_df=util.average_results_df(results_df)
		return results_df