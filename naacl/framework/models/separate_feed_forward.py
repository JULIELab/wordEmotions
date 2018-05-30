import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.stats as st
from sklearn.model_selection import KFold
import naacl.framework.util as util
'''
Input and output layers are shared. The hidden layers are specific for
the respective dimensions.
'''

def define_model(layers, nonlinearity=tf.sigmoid,  weights_sd=0.01, biases=0):
	'''
	layers				int list indicating the number of neurons per layer
						(including input and output layer)
	'''

	dropout_embedding=tf.placeholder(tf.float32, name='dropout_embedding')
	dropout_hidden=tf.placeholder(tf.float32, name='dropout_hidden')



	#input layer 
	input_layer=tf.nn.dropout(tf.placeholder(tf.float32, shape=[None,layers[0]], name='input_layer'),
							  keep_prob=1.-dropout_embedding)

	top_level_activation=[None]*layers[-1]
	outputs=[None]*layers[-1]
	#each "branch"
	for branch in range(layers[-1]):
		activations=[None]*(len(layers)-1)
		activations[0]=input_layer
		#create branch
		for i in np.arange(start=1, stop=len(activations), step=1):
			w_i, b_i, z_i=util.get_layer(shape=[layers[i-1], layers[i]],
									activation=activations[i-1],
									weights_sd=weights_sd,
									biases=biases)
			a_i=nonlinearity(z_i)
			a_i_drop=tf.nn.dropout(a_i, keep_prob=1.-dropout_hidden,
							   name='hidden_activation')
			activations[i]=a_i_drop
			if i==len(activations)-1:
				top_level_activation[branch]=activations[-1]
	#create output layer
	for i in range(len(outputs)):
		w,b,z=util.get_layer(shape=[layers[-2],1],
							 activation=top_level_activation[i],
							 weights_sd=weights_sd,
							 biases=biases)
		outputs[i]=z
	output_layer=tf.concat(outputs, axis=1, name='output_layer')



def define_loss(loss_function=tf.losses.mean_squared_error,
				l2_beta=0):
	output_layer=tf.get_default_graph().get_tensor_by_name('output_layer:0') #':0' makes it refer to the tensor instead of the operation!!!
	y=tf.placeholder(tf.float32, shape=[None,output_layer.shape[1]], name='actual_values')
	if l2_beta>0:
		regulizer=tf.reduce_mean([tf.nn.l2_loss(w) for w \
						in tf.get_collection(tf.GraphKeys.WEIGHTS)])
		# print(regulizer)
		loss=loss_function(y, output_layer)+(l2_beta*regulizer)
		tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

	else:	
		loss=loss_function(y, output_layer)
	# print(tf.get_collection(tf.GraphKeys.WEIGHTS))
	# print(tf.get_collection(tf.GraphKeys.LOSSES))
	# util.print_nodes()

def define_optimization(learning_rate=1e-2):
	loss=tf.get_collection(tf.GraphKeys.LOSSES)[-1] #get last entry in loss collection (i.e., the last one added to the collection)
	optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')
	training_objective=optimizer.minimize(loss, name='training_objective')

def init_session(session):
	init=tf.global_variables_initializer() # name: 'init'
	session.run(init)

def train(session,
		  features,
		  labels, 
		  training_steps, 
		  batch_size,
		  dropout_hidden,
		  dropout_embedding,
		  batch_gen=util.Batch_Gen,
		  report_at=0):
	'''
	Standard training function using dropout (on embedding and hidden layers).
	'''
	sess=session
	loss=tf.get_collection(tf.GraphKeys.LOSSES)[-1]
	training_objective=tf.get_default_graph().get_operation_by_name('training_objective')
	batch_gen=batch_gen(features, labels, batch_size)
	for i_step in range(training_steps):
		curr_features, curr_labels=batch_gen.next()
		# monitor training
		if report_at>0 and i_step%report_at==0:
			curr_loss,preds=sess.run([loss,'output_layer:0'],
							 feed_dict={'input_layer:0':curr_features,
									    'dropout_embedding:0':.0,
									    'dropout_hidden:0':.0,
									    'actual_values:0':curr_labels})
			preds=pd.DataFrame(data=preds, columns=curr_labels.columns, index=curr_labels.index)
			perf=util.eval(true=curr_labels, prediction=preds)
			print(i_step, curr_loss, np.mean(perf))
		#acutal training step
		sess.run(training_objective,
				 feed_dict={'input_layer:0':curr_features,
						    'dropout_embedding:0':dropout_embedding,
						    'dropout_hidden:0':dropout_hidden
						    ,
						    'actual_values:0':curr_labels})

def test(session, features, labels):
	sess=session
	preds=sess.run('output_layer:0',
							 feed_dict={'input_layer:0':features,
									    'dropout_embedding:0':.0,
									    'dropout_hidden:0':.0})
	preds=pd.DataFrame(data=preds, columns=labels.columns, index=labels.index)
	return util.eval(labels, preds)

def crossvalidate(features, 
			  labels, 
			  k_folds, 
			  layers, 
			  nonlinearity, 
			  loss_function,
			  l2_beta,
			  learning_rate, 
			  training_steps, 
			  batch_size, 
			  dropout_hidden, 
			  dropout_embedding, 
			  report_at=50,
			  weights_sd=.01,
			  biases=0):
	k=0
	results_df=pd.DataFrame(columns=labels.columns)
	for fold in util.k_folds_split(features, labels, k=k_folds):
		tf.reset_default_graph()
		k+=1
		print(k)
		features_train, labels_train, features_test, labels_test=fold
		# define model
		define_model(layers=layers,
						   nonlinearity=nonlinearity,
						   weights_sd=weights_sd,
						   biases=biases)
		define_loss(loss_function=tf.losses.mean_squared_error,
				  l2_beta=0)
		define_optimization(learning_rate=learning_rate)
		with tf.Session() as sess:
			init_session(sess)
			train(session=sess, 
						features=features_train, 
						labels=labels_train, 
						training_steps=training_steps, 
						batch_size=batch_size,
						dropout_hidden=dropout_hidden,
						dropout_embedding=dropout_embedding, 
						report_at=report_at)
			results_df.loc[k]=test(sess, 
										 features_test, 
										 labels_test)
	results_df=util.average_results_df(results_df)
	return results_df

def test_at_steps(features,
				  labels,
				  test_size,
				  steps,
				  runs,
				  layers,
				  nonlinearity,
				  loss_function,
				  l2_beta,
				  learning_rate,
				  batch_size,
				  dropout_hidden,
				  dropout_embedding,
				  report_at=0,
				  weights_sd=.01,
				  biases=0.,
				  batch_gen=util.Serial_Batch_Gen):
	'''
	Trains a model and repetitively tests its performance and given numbers of
	training steps. The process is repeated multiple times and the results are 
	averaged.

	Args:
	steps...........Int list; the number of training steps after which
					performance will be tested
	runs............Int; the number of times this will be repeated and then
					averaged. For each run, a random sample (with replacement)
					will be drawn from <labels>
	test_size.......Float; The share of the test data.
	'''

	data_frames_from_runs={}

	for i_run in range(runs):
		print('Run: ',i_run)
		features_train,\
		features_test,\
		labels_train,\
		labels_test=util.train_test_split(features, labels, test_size)
		# create model
		tf.reset_default_graph()
		define_model(	layers=layers, 
					 	nonlinearity=nonlinearity,
					 	weights_sd=weights_sd,
					 	biases=biases)
		define_loss(loss_function=loss_function, l2_beta=l2_beta)
		define_optimization(learning_rate=learning_rate)

		results_df=pd.DataFrame(columns=list(labels)+['Average'], index=steps)
		total_steps=0
		with tf.Session() as sess:
			init_session(sess)
			for i_step in range(len(steps)):
				# determine the number of next training steps
				# print(i_step)
				if total_steps==0 and i_step==0:
					next_steps=steps[0]
				else:
					next_steps=steps[i_step]-steps[i_step-1]
				train(session=sess,
					  features=features_train,
					  labels=labels_train, 
					  training_steps=next_steps,
					  batch_gen=batch_gen,
					  batch_size=batch_size,
					  dropout_hidden=dropout_hidden,
					  dropout_embedding=dropout_embedding,
					  report_at=report_at)

				total_steps+=next_steps

				curr_results=test(session=sess,
								  features=features_test,
								  labels=labels_test)
				curr_row=np.mean(curr_results)

				results_df.loc[total_steps]=curr_results+[curr_row]
				# print(results_df)
				print(steps[i_step],curr_row)

		data_frames_from_runs[i_run]=results_df
	# average all data frames cell-wise
	panel=pd.Panel(data_frames_from_runs)
	averaged_df=panel.mean(axis=0)
	sd_df=panel.std(axis=0)
	column_names=[]
	for col in list(averaged_df):
		column_names+=[col+'.M', col+'.SD']
	out_df=pd.DataFrame(columns=column_names, index=averaged_df.index)
	for var in list(averaged_df):
		out_df[var+'.M']=averaged_df[var]
		out_df[var+'.SD']=sd_df[var]
	return out_df



