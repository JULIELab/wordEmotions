import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from naacl.framework import util
import os

results='results/'


# change font size
matplotlib.rcParams.update({'font.size': 12})


f=plt.figure()

#anew joint
df=util.load_tsv(results+'anew_joint.tsv')
plt.plot(df.index, df['Average.M'], label='MTLNN on EN', color='blue',
			linestyle='-',
			marker='o')

#anew separate
df=util.load_tsv(results+'anew_separate.tsv')
plt.plot(df.index, df['Average.M'], label='sepNN on EN', color='blue',
			linestyle='--',
			marker='o')


#warriner joint
df=util.load_tsv(results+'warriner_joint.tsv')
plt.plot(df.index, df['Average.M'], label='MTLNN on EN+', color='red',
			linestyle='-',
			marker='^')

#warriner separate
df=util.load_tsv(results+'warriner_separate.tsv')
plt.plot(df.index, df['Average.M'], label='sepNN on EN+', color='red',
			linestyle='--',
			marker='^')


# for file in os.listdir(results):
# 	if file.endswith('.tsv'):
# 		print(file)
# 		df=util.load_tsv(results+file)
# 		# plt.plot([0.]+list(df.index), [0.]+list(df['Average.M']), label=file[:-4])
# 		plt.plot(list(df.index), list(df['Average.M']), label=file[:-4])

# plt.ylim([.70, .85])
plt.xlabel('Training Steps')
plt.ylabel('Pearson Correlation')
plt.grid()
plt.legend(borderpad=1)
f.savefig('plot.pdf')