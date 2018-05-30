import naacl.framework.prepare_data as data
import naacl.framework.util as util
from sklearn.model_selection import train_test_split as split
import scipy.stats as st
import naacl.framework.util as util
from sklearn.model_selection import KFold

embs=data.get_google_sgns(vocab_limit=50000)
# labels=data.load_anew99()
labels=data.load_anew99()

import naacl.framework.reference_methods.densifier as densifier


ds=densifier.Densifier(embs)
train,test=split(labels, test_size=.1)
print(ds.crossvalidate(labels,k_folds=2))


