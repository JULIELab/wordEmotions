import pandas as pd 
from naacl.framework import util


'''
Queries the results of the parameter grid search and presents it as a 
sorted table.

===> 	Threshold of .5 SDs seems to be most appropriate. Alpha may be less 
		important with .3 to .9 all being roughly identical in performance.
		.7 however was best and will be employed in the future
'''


meta=util.load_tsv('results/meta.tsv')
# print(meta)
for i in meta.index:
	tmp=util.load_tsv(i)
	meta.loc[i, 'Average_Performance']=tmp.loc['Average', 'Average']

meta=meta.sort_values(by='Average_Performance', ascending=False)

print(meta)
util.save_tsv(meta, 'grid_search_results.tsv')