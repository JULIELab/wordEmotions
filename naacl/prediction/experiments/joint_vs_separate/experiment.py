import pandas as pd 
from naacl.framework import util
from naacl.framework import prepare_data as data
from naacl.framework.models import joint_feed_forward as joint 
from naacl.framework.models import separate_feed_forward as separate
import os
import tensorflow as tf

anew=data.load_anew99()
warriner=data.load_warriner13()

embs=data.get_facebook_fasttext_common_crawl()
# embs=data.get_google_sgns(vocab_limit=50000)
features_anew=util.feature_extraction(anew.index,embs)
features_warriner=util.feature_extraction(warriner.index, embs)

settings={	'anew_joint':{	'lexicon':anew, 
							'features':features_anew,
							'model':joint
							},
			'anew_separate':{	'lexicon':anew, 
								'features':features_anew,
								'model':separate
								},
			'warriner_joint':{	'lexicon':warriner, 
								'features':features_warriner,
								'model':joint
								},
			'warriner_separate':{	'lexicon':warriner, 
									'features':features_warriner,
									'model':separate
								},
			}

if not os.path.isdir('results'):
	os.makedirs('results')



for setting_name in settings:
	print('Now processing {}'.format(setting_name))
	setting=settings[setting_name]
	model=setting['model']
	labels=setting['lexicon']
	features=setting['features']
	result= model.test_at_steps(  features=features,
						  labels=labels,
						  test_size=.1,
						  steps=list(range(1000,15000+1, 1000)),
						  runs=20,
						  layers=[len(list(features)), 256, 128,
									len(list(labels))],
						  nonlinearity=util.leaky_relu,
						  loss_function=tf.losses.mean_squared_error,
						  l2_beta=0,
						  learning_rate=1e-3,
						  batch_size=128,
						  dropout_hidden=.5,
						  dropout_embedding=.2,
						  report_at=0,
						  weights_sd=.001,
						  biases=.01,
						  batch_gen=util.Serial_Batch_Gen)
	util.save_tsv(result, 'results/'+setting_name+'.tsv')

