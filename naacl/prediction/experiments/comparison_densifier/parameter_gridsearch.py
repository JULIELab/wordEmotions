import naacl.framework.prepare_data as data
import naacl.framework.util as util
from sklearn.model_selection import train_test_split as split
import scipy.stats as st
import naacl.framework.util as util
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

embs=data.get_google_sgns(vocab_limit=None)
labels=data.load_anew99()

import naacl.framework.reference_methods.densifier as densifier



ds=densifier.Densifier(embs)

configs=[[threshold,alpha] for threshold in np.arange(start=0, stop=1.5000001,step=.5)\
			for alpha in np.arange(start=.1, stop=.90000001, step=.2)]


print(configs)

results_config={str(config):pd.DataFrame(columns=labels.columns) for\
					config in configs}


k=0
for  train_index, test_index in KFold(n_splits=10, shuffle=True).\
			split(labels):
	k+=1
	train=labels.iloc[train_index]
	test=labels.iloc[test_index]
	print(k)

	for config in configs:
		print(config)

		threshold=config[0]
		alpha=config[1]

		ds.fit(	seed_lexicon=train,
				binarization_threshold=threshold,
				alpha=alpha)
		prediction=ds.predict(words=test.index)
		performance=util.eval(test, prediction)
		print(performance)
		results_config[str(config)].loc[k]=performance

meta_df=pd.DataFrame(columns=['threshold', 'alpha'])

for config in configs:
	results_df=results_config[str(config)]
	results_df=util.average_results_df(results_df)
	fname='results/{}.tsv'.format(str(config))
	util.save_tsv(results_df, fname)
	meta_df.loc[fname]=config

util.save_tsv(meta_df, 'results/meta.tsv')



