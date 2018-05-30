import naacl.framework.models.joint_feed_forward as model
import naacl.framework.util as util
import tensorflow as tf
import time

SETTINGS={
	# 'layers':[5, 128, 128, 3],
	'batch_size':128,
	'weights_sd':.001,
	'biases':.01,
	'loss_function':tf.losses.mean_squared_error,
	'l2_beta':0,
	'learning_rate':1e-3,
	'k_folds':10,
	'training_steps':15000,
	'dropout_hidden':.5,
	'dropout_embedding':.2,
	'nonlinearity':util.leaky_relu,
	'batch_gen':util.Serial_Batch_Gen
}



def fit(features, labels):
	tf.reset_default_graph()
	model.define_model(	layers=[len(list(features)), 256, 128,
									len(list(labels))],
						nonlinearity=SETTINGS['nonlinearity'],
						weights_sd=SETTINGS['weights_sd'],
						biases=SETTINGS['biases'])
	model.define_loss(	loss_function=SETTINGS['loss_function'],
						l2_beta=SETTINGS['l2_beta'])
	model.define_optimization(learning_rate=SETTINGS['learning_rate'])
	###
	session=tf.Session()
	###
	model.init_session(session)
	model.train(	session=session,
					features=features,
					labels=labels,
					training_steps=SETTINGS['training_steps'],
					batch_gen=SETTINGS['batch_gen'],
					batch_size=SETTINGS['batch_size'],
					dropout_hidden=SETTINGS['dropout_hidden'],
					dropout_embedding=SETTINGS['dropout_embedding'],
					report_at=250)
	return session


def predict(features, session, var_names):
	return model.predict(features, session, var_names)

def evaluate(train_features, train_labels, test_features, test_labels):
	return model.evaluate(	train_features=train_features,
							train_labels=train_labels,
							test_features=test_features,
							test_labels=test_labels,
							runs=10,
							layers=[len(list(train_features)), 256, 128,
									len(list(train_labels))],
							nonlinearity=SETTINGS['nonlinearity'],
							loss_function=SETTINGS['loss_function'],
							l2_beta=SETTINGS['l2_beta'],
							learning_rate=SETTINGS['learning_rate'],
							training_steps=SETTINGS['training_steps'],
							batch_size=SETTINGS['batch_size'],
							dropout_embedding=SETTINGS['dropout_embedding'],
							dropout_hidden=SETTINGS['dropout_hidden'],
							weights_sd=SETTINGS['weights_sd'],
							biases=SETTINGS['biases'],
							report_at=1000)

def crossvalidate(	features, 
			  		labels):
	return model.crossvalidate(	features=features, 
					  		labels=labels, 
							k_folds=SETTINGS['k_folds'],  
							layers=[len(list(features)), 256, 128,
									len(list(labels))],
							nonlinearity=SETTINGS['nonlinearity'], 
							loss_function=SETTINGS['loss_function'],
							l2_beta=SETTINGS['l2_beta'],
							learning_rate=SETTINGS['learning_rate'], 
							training_steps=SETTINGS['training_steps'], 
							batch_size=SETTINGS['batch_size'], 
							dropout_hidden=SETTINGS['dropout_hidden'], 
							dropout_embedding=SETTINGS['dropout_embedding'], 
							report_at=1000,
							weights_sd=SETTINGS['weights_sd'],
							biases=SETTINGS['biases'])

def test_at_steps(features, labels):
	return model.test_at_steps(	features=features,
							  	labels=labels,
							  	test_size=.1,
							  	steps=list(range(1000,SETTINGS['training_steps']+1, 1000)),
							  	runs=10,
							  	layers=[len(list(features)), 256, 128,
									len(list(labels))],
							  	nonlinearity=SETTINGS['nonlinearity'],
							  	loss_function=SETTINGS['loss_function'],
							  	l2_beta=SETTINGS['l2_beta'],
								learning_rate=SETTINGS['learning_rate'], 
								batch_size=SETTINGS['batch_size'], 
								dropout_hidden=SETTINGS['dropout_hidden'], 
								dropout_embedding=SETTINGS['dropout_embedding'], 
							  	report_at=0,
							  	weights_sd=SETTINGS['weights_sd'],
								biases=SETTINGS['biases'])


