import pandas as pd
import numpy as np 
import os
from naacl.framework import util
from scipy import stats as st

RESULTS='../results/'

setups=[]


for setup in os.listdir(RESULTS):
	if os.path.isdir(RESULTS+setup):
		setups.append(setup)

# print(setups)

table=pd.DataFrame(index=setups)

for setup in table.index:
	path=RESULTS+setup
	methods=os.listdir(path)
	for method in methods:
		if method.endswith('.tsv'):
			avg=util.get_average_result_from_df(path+'/'+method)
			# print(avg)
			table.loc[setup,method[:-4]]=avg

### reordering and dropping columns
table=table[['linear_model', 'li_regressor', 'turney', 'densifier', 'aicyber', 'my_model_relu']]


### reordering rows
table=table.reindex([
			   'english_warriner_google',
			   'english_warriner_common_crawl',
			   'english_warriner_wikipedia',
			   'english_warriner_sedoc',
			   'english_anew_google',
			   'english_anew_common_crawl',
			   'english_anew_wikipedia',
			   'spanish_redondo_wikipedia',
			   'spanish_stadthagen_wikipedia',
			   'german_schmidtke_wikipedia',
			   'chinese_yu_wikipedia',
			   'polish_imbir_wikipedia',
			   'italian_montefinese_wikipedia',
			   'portuguese_soares_wikipedia',
			   'dutch_moors_wikipedia',
			   'indonesian_sianipar_wikipedia'
			   ])



### Perform test if best system significantly outperforms the second 
### (paired, two-tailed t-test; p < .05)

print()
#for each row, identify best and second best system
sig_table=pd.DataFrame(index=table.index, columns=['first', 'second', 'p-value', 'sig-level'])
for setup in sig_table.index:
	sorted=table.loc[setup,:].sort_values(ascending=False)
	best2=list(sorted.index[:2])
	# retrieve data frames
	first=util.load_tsv(RESULTS+setup+'/'+best2[0]+'.tsv')
	second=util.load_tsv(RESULTS+setup+'/'+best2[1]+'.tsv')
	#compute p-values based on averages over VA(D)
	siglevel=''
	pvalue=st.ttest_rel(a=first['Average'], b=second['Average'])[1]
	print(pvalue)
	if pvalue >=.05:
		siglevel='–'
	elif pvalue <.05 and pvalue >=.01:
		siglevel='*'
	elif pvalue <.01 and pvalue >=.001:
		siglevel='**'
	else:
		siglevel='***'
	# fill data frame
	sig_table.loc[setup]=best2+[pvalue,siglevel]


print(sig_table)
util.save_tsv(sig_table, 'significance_test_results.tsv')



table.columns=['LinReg', 'RidgReg', 'TL', 'Densifier',
			   'ensembleNN', 'jointNN']



print('\nt-test for averages: ')
print(st.ttest_rel(a=table.ensembleNN, b=table.jointNN))
print()

print()
table.loc['Average']=table.mean(axis=0)
# print(table)
util.save_tsv(table, 'overall_table.tsv')
print(table.to_latex(float_format="%.3f"))




