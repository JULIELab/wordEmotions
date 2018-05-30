import numpy as np 
import tensorflow as tf 
import pandas as pd 
# import naacl.framework.constants as cs 
import naacl.framework.prepare_data as data
import naacl.framework.util as util
# import naacl.framework.representations.embedding as embedding
from sklearn.model_selection import KFold
import os
import sklearn.linear_model
import scipy.stats as st
import sys

class Setting():
	def __init__(self, name, data, embeddings):
		'''
		Args:
			name 			String (describing results_file)
			data 			function
			embeddings 		function
		'''
		self.name=name
		self.load_data=data
		self.load_embeddings=embeddings



SETTINGS=[	# English Anew, 3 different embeddings
			Setting('english_anew_google', 
					data.load_anew99, 
					lambda:data.get_google_sgns(vocab_limit=None)),
			Setting('english_anew_common_crawl',
					data.load_anew99,
					data.get_facebook_fasttext_common_crawl),
			Setting('english_anew_wikipedia',
					data.load_anew99,
					lambda:data.get_facebook_fasttext_wikipedia('english', vocab_limit=None)),
			#English Warriner, 3 different embeddings
			Setting('english_warriner_google',
					data.load_warriner13,
					lambda:data.get_google_sgns(vocab_limit=None)),
			Setting('english_warriner_common_crawl',
					data.load_warriner13,
					data.get_facebook_fasttext_common_crawl),
			Setting('english_warriner_wikipedia',
					data.load_warriner13,
					lambda:data.get_facebook_fasttext_wikipedia('english')),
			### comparison with sedoc
			Setting('english_warriner_sedoc',
					data.load_warriner13,
					data.get_sedoc17_embeddings),
			# Germanic
			Setting('german_schmidtke_wikipedia',
					lambda:data.load_schmidtke14(lower_case=True),
					lambda:data.get_facebook_fasttext_wikipedia('german')),
			Setting('dutch_moors_wikipedia',
					data.load_moors13,
					lambda:data.get_facebook_fasttext_wikipedia('dutch')),
			# Romance
			Setting('spanish_redondo_wikipedia',
					data.load_redondo07,
					lambda:data.get_facebook_fasttext_wikipedia('spanish')),
			Setting('spanish_stadthagen_wikipedia',
					data.load_stadthagen16,
					lambda:data.get_facebook_fasttext_wikipedia('spanish')),
			Setting('italian_montefinese_wikipedia',
					data.load_montefinese14,
					lambda:data.get_facebook_fasttext_wikipedia('italian')),
			Setting('portuguese_soares_wikipedia',
					data.load_soares12,
					lambda:data.get_facebook_fasttext_wikipedia('portuguese')),
			# others
			Setting('polish_imbir_wikipedia',
					data.load_imbir16,
					lambda:data.get_facebook_fasttext_wikipedia('polish')),

			Setting('chinese_yu_wikipedia',
					data.load_yu16,
					lambda:data.get_facebook_fasttext_wikipedia('chinese')),
			#
			Setting('indonesian_sianipar_wikipedia',
					data.load_sianipar16,
					lambda:data.get_facebook_fasttext_wikipedia('indonesian'))
		]


import naacl.prediction.experiments.main.my_model_relu as my_model_relu
import naacl.prediction.experiments.main.my_model_sigmoid as my_model_sigmoid
from naacl.framework.reference_methods import aicyber, densifier,turney
from naacl.framework.reference_methods.li import Multi_Target_Regressor as li_regressor



def main(results_path='results', metric='r'):
	RESULTS=results_path+'/'

	if not os.path.exists(RESULTS):
	    os.makedirs(RESULTS)

	### settings
	for setting in SETTINGS:
		print('Now processing {}'.format(setting.name))

		### check if this setting has already been processed
		if os.path.isdir(RESULTS+setting.name):
			print('\t{} has already been processed!'.format(setting.name))
		else:


			labels=setting.load_data()
			embs=setting.load_embeddings()

			models={
					'turney':turney.Bootstrapper(embs), 
					'densifier':densifier.Densifier(embs), 
					'my_model_relu':my_model_relu,
					'my_model_sigmoid':my_model_sigmoid, 
					'aicyber':aicyber.mlp_ensemble(), 
					'li_regressor':li_regressor(),
					'linear_model':li_regressor(
						init_fun=sklearn.linear_model.LinearRegression)
					}

			results_setting={key:pd.DataFrame(columns=labels.columns)\
				for key in list(models)}


			### Crossvalidation
			k=0
			for  train_index, test_index in KFold(n_splits=10, shuffle=True).\
						split(labels):
				k+=1
				train=labels.iloc[train_index]
				test=labels.iloc[test_index]
				print(k)

				train_features=util.feature_extraction(train.index, embs)
				test_features=util.feature_extraction(test.index, embs)

				### methods
				for model_name in list(models):
					model=models[model_name]
					print(model_name)

					### case distinction because models do not share the same
					###	interface 
					tf.reset_default_graph()
					preds=None
					if model_name in ['aicyber', 'li_regressor', 'linear_model']:
						model.fit(train_features.copy(), train.copy())
						preds=model.predict(test_features.copy())
					elif model_name in ['my_model_relu', 'my_model_sigmoid']:
						# print(train)
						# sess=tf.Session()
						session=model.fit(train_features.copy(), train.copy())
						preds=model.predict(test_features.copy(), 
							session, var_names=train.columns)
						del session
					else:
						model.fit(train.copy())
						preds=model.predict(test.index.copy())
						###
						print(test)
						print(preds)
						###
					perf=util.eval(test,preds, metric)
					print(perf)
					results_setting[model_name].loc[k]=perf
					print(results_setting[model_name])

			
			os.makedirs(RESULTS+setting.name)
			### after cv, for each individual results data frame, average results and save data
			for model_name in list(models):
				curr_results=results_setting[model_name]
				curr_results=util.average_results_df(curr_results)
				fname='{}{}/{}.tsv'.format(RESULTS, setting.name, model_name)
				util.save_tsv(curr_results, fname)
			print('\tFinished processing {}'.format(setting.name))

			### delete respective setting to free up memory
		del setting


if __name__=='__main__':
	main(results_path=sys.argv[1], metric=sys.argv[2])


