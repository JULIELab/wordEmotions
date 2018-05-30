import os
import numpy as np
import pandas as pd
import naacl.framework.constants as cs
from io import StringIO
from naacl.framework.representations.embedding import Embedding
from naacl.framework.util import scaleInRange
from naacl.framework.util import duplicate

heads_vad = ['Word','Valence','Arousal','Dominance']
heads_be5 = ['Word','Joy','Anger','Sadness','Fear','Disgust']



#### ENGLISH

def load_anew10():
	anew = pd.read_csv(cs.anew10, sep = '\t')
	anew = anew[['Word','ValMn','AroMn','DomMn']]
	anew.columns = ['Word', 'Valence', 'Arousal',
				   'Dominance']
	anew.set_index('Word', inplace=True)
	return anew

def load_anew99():
	anew=pd.read_csv(cs.anew99, sep='\t')
	anew.columns=heads_vad
	anew.set_index('Word', inplace=True)
	#schmidtke14=schmidtke14[~schmidtke14.index.duplicated(keep='first')]
	#drop duplicates
	#anew=anew[~anew.index.duplicated(keep='first')]
	anew=duplicate(anew)
	return anew

#print(anew.head())

def load_stevenson07():
	stevenson07=pd.read_excel(cs.stevenson07)
	stevenson07=stevenson07[['word','mean_hap','mean_ang','mean_sad',
							 'mean_fear','mean_dis']]
	stevenson07.columns=['Word', 'Joy','Anger','Sadness','Fear','Disgust']
	#TD: stevenson07.columns=heads_be5
	stevenson07.set_index('Word', inplace=True)
	#print(stevenson07.head())
	#print(type(stevenson07.Joy[1]))
	return stevenson07



def load_warriner13():
	warriner13 = pd.read_csv(cs.warriner13, sep=',')
	warriner13=warriner13[['Word','V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']]
	warriner13.columns=heads_vad
	warriner13.set_index('Word',inplace=True)
	#print(warriner13.head())
	#print(warriner13.shape)
	return warriner13



# #### SPANISH
def load_redondo07():
	redondo07=pd.read_excel(cs.redondo07)
	redondo07=redondo07[['S-Word','Val-Mn-All','Aro-Mn-All','Dom-Mn-All']]
	redondo07.columns = heads_vad
	redondo07.set_index('Word', inplace=True)
	#print(redondo07.head())
	#print(redondo07.shape)
	return redondo07



def load_ferre16():
	ferre16=pd.read_excel(cs.ferre16)
	ferre16=ferre16[['Spanish_Word','Hap_Mean','Ang_Mean','Sad_Mean',
					 'Fear_Mean','Disg_Mean']]
	ferre16.columns=heads_be5
	ferre16.set_index('Word', inplace=True)
	#print(ferre16.head())
	#print(ferre16.shape)
	return ferre16



# #### POLISH

def load_riegel15():
	riegel15=pd.read_excel(cs.riegel15)
	riegel15=riegel15[['NAWL_word','val_M_all','aro_M_all']]
	riegel15.columns=['Word','Valence','Arousal']
	riegel15.set_index('Word', inplace=True)
	return riegel15



def load_wierzba15():
	wierzba15 = pd.read_excel(cs.wierzba15)
	wierzba15=wierzba15[['NAWL_word', 'hap_M_all', 'ang_M_all', 'sad_M_all',
						 'fea_M_all', 'dis_M_all']]
	wierzba15.columns=heads_be5
	wierzba15.set_index('Word', inplace=True)
	## rescaling basic emotions


	## Scaling
	for cat in ['Joy', 'Anger', 'Sadness', 'Fear', 'Disgust']:
		wierzba15[cat] = [scaleInRange(x=x, oldmin=1.,
											 oldmax=7., newmin=1., newmax=5.) 
								for x in wierzba15[cat]]


	# print(wierzba15.head())
	# print(wierzba15.shape)
	return wierzba15


def load_imbir16():
	imbir16 = pd.read_excel(cs.imbir16)
	imbir16 = imbir16[['polish word', 'Valence_M', 'arousal_M', 'dominance_M']]
	imbir16.columns=heads_vad
	imbir16.set_index('Word', inplace=True)
	# print(imbir16.head())
	# print(imbir16.shape)
	return imbir16



# ### GERMAN

def load_schmidtke14(lower_case=False):
	schmidtke14=pd.read_excel(cs.schmidtke14)
	# schmidtke14=schmidtke14[['Word','Valence','Arousal','Dominance']]
	schmidtke14=schmidtke14[['G-word', 'VAL_Mean', 'ARO_Mean_(ANEW)', 'DOM_Mean']]
	schmidtke14.columns=['Word', 'Valence', 'Arousal', 'Dominance']
	#TD: schmidtke14.columns=heads_vad
	# schmidtke14['Word']=schmidtke14['Word'].str.lower()
	schmidtke14.set_index('Word', inplace=True)

	if lower_case:
		schmidtke14.index=schmidtke14.index.str.lower()

	#schmidtke14=schmidtke14[~schmidtke14.index.duplicated(keep='first')]
	schmidtke14=duplicate(schmidtke14)
	## rescaling valence
	# schmidtke14=pd.read_csv(cs.schmidtke14, sep='\t')
	# schmidtke14=schmidtke14[['Word','Valence','Arousal','Dominance']]
	# schmidtke14.columns=heads_vad
	# schmidtke14['Word']=schmidtke14['Word'].str.lower()
	# schmidtke14.set_index('Word', inplace=True)
	# ## rescaling valence

	schmidtke14.Valence = [scaleInRange(x = x, oldmin = -3.,
									   oldmax = 3., newmin = 1., newmax=9.) 
						   for x in schmidtke14.Valence]

	# ### setting word column to lower case for compatiblity with briesemeister11
	# # print(schmidtke14.head())
	# # print(schmidtke14.shape)

	return schmidtke14

def load_briesemeister11():
	briesemeister11=pd.read_excel(cs.briesemeister11)
	briesemeister11=briesemeister11[['WORD_LOWER', 'HAP_MEAN', 'ANG_MEAN',
									 'SAD_MEAN', 'FEA_MEAN', 'DIS_MEAN']]
	briesemeister11.columns=heads_be5
	briesemeister11.set_index('Word', inplace=True)
	# print(briesemeister11.head())
	# print(briesemeister11.shape)
	return briesemeister11



def load_hinojosa16():
	hinojosa16a=pd.read_excel(cs.hinojosa16a)
	hinojosa16a=hinojosa16a[['Word','Val_Mn', 'Ar_Mn', 'Hap_Mn', 'Ang_Mn','Sad_Mn',
							 'Fear_Mn', 'Disg_Mn']]
	hinojosa16a.columns=['Word', 'Valence', 'Arousal',
						 'Joy','Anger','Sadness','Fear','Disgust']
	hinojosa16a.set_index('Word', inplace=True)
	hinojosa16b=pd.read_excel(cs.hinojosa16b)
	hinojosa16b=hinojosa16b[['Word', 'Dom_Mn']]
	hinojosa16b.columns=['Word','Dominance']
	hinojosa16b.set_index('Word', inplace=True)
	hinojosa=hinojosa16a.join(hinojosa16b, how='inner')
	hinojosa=hinojosa[['Valence', 'Arousal', 'Dominance',
			
					   'Joy', 'Anger', 'Sadness', 'Fear', 'Disgust']]
	return hinojosa


def load_stadthagen16():
	stadthagen16=pd.read_csv(cs.stadthagen16, encoding='cp1252')
	stadthagen16=stadthagen16[['Word', 'ValenceMean', 'ArousalMean']]
	stadthagen16.columns=['Word', 'Valence', 'Arousal']
	stadthagen16.set_index('Word', inplace=True)
	return stadthagen16

def load_kanske10():
	with open(cs.kanske10, encoding='cp1252') as f:
		kanske10=f.readlines()
	# Filtering out the relevant portion of the provided file
	kanske10=kanske10[7:1008]
	# Creating data frame from string: 
	#https://stackoverflow.com/questions/22604564/how-to-create-a-pandas-dataframe-from-string
	kanske10=pd.read_csv(StringIO(''.join(kanske10)), sep='\t')
	kanske10=kanske10[['word', 'valence_mean','arousal_mean']]
	kanske10.columns=['Word', 'Valence', 'Arousal']
	kanske10['Word']=kanske10['Word'].str.lower()
	kanske10.set_index('Word', inplace=True)
	return kanske10
# kanske = load_kanske10()
# print(kanske)
# for i in kanske.columns:
# 	print(type(kanske[i].iloc[1]))

def load_guasch15():
	guasch15=pd.read_excel(cs.guasch15)
	guasch15=guasch15[['Word','VAL_M', 'ARO_M']]
	guasch15.columns=['Word', 'Valence', 'Arousal']
	guasch15.set_index('Word', inplace=True)
	return guasch15

def load_moors13():
	# with open(cs.moors13) as f:
	# 	moors13=f.readlines()
	# moors13=moors13[1:]
	# moors13=pd.read_excel(StringIO(''.join(moors13)))
	moors13=pd.read_excel(cs.moors13, header=1)
	moors13=moors13[['Words', 'M V', 'M A', 'M P']]
	moors13.columns=heads_vad
	moors13.set_index('Word', inplace=True)
	# print(moors13)
	return moors13

def load_montefinese14():
	montefinese14=pd.read_excel(cs.montefinese14, header=1)
	montefinese14=montefinese14[['Ita_Word', 'M_Val', 'M_Aro', 'M_Dom']]
	montefinese14.columns=heads_vad
	montefinese14.set_index('Word', inplace=True)
	return montefinese14

def load_soares12():
	soares12=pd.read_excel(cs.soares12, sheetname=1)
	soares12=soares12[['EP-Word', 'Val-M', 'Arou-M', 'Dom-M']]
	soares12.columns=heads_vad
	soares12.set_index('Word', inplace=True)
	return soares12

def load_sianipar16():
	sianipar16=pd.read_excel(cs.sianipar16)
	sianipar16=sianipar16[['Words (Indonesian)', 'ALL_Valence_Mean', 'ALL_Arousal_Mean', 'ALL_Dominance_Mean']]
	sianipar16.columns=heads_vad
	sianipar16.set_index('Word', inplace=True)
	#sianipar16=sianipar16[~sianipar16.index.duplicated(keep='first')]
	sianipar16=duplicate(sianipar16)
	return sianipar16

def load_yu16():
	'''
	Yu, L.-C., Lee, L.-H., Hao, S., Wang, J., He, Y., Hu, J., … Zhang, X. 
	(2016). Building Chinese Affective Resources in Valence-Arousal Dimensions.
	In Proceedings of NAACL-2016.
	'''
	yu16=pd.read_csv(cs.yu16)
	yu16=yu16[['Word', 'Valence_Mean', 'Arousal_Mean']]
	yu16.columns=heads_vad[:-1]
	yu16.set_index('Word', inplace=True)
	return yu16

def load_yu16_ialp_train_test():
	train=pd.read_csv(cs.yu16)
	train=train[['No.', 'Word', 'Valence_Mean', 'Arousal_Mean']]
	train.columns=[['id', 'Word', 'Valence', 'Arousal']]
	# train.set_index('id', inplace=True)
	test=train.copy()
	train=train.loc[train.id.isin(range(1,1654))]
	test=test.loc[test.id.isin(range(1654,2150))]
	def __format__(df):
		return df[['Word', 'Valence', 'Arousal']].set_index('Word')

	test=__format__(test)
	train=__format__(train)
	return train,test

def load_yao16():
	'''
	Yao, Z., Wu, J., Zhang, Y., & Wang, Z. (2016). Norms of valence, arousal, 
	concreteness, familiarity, imageability, and context availability for 
	1,100 Chinese words. Behavior Research Methods.
	'''
	with open(cs.yao16) as f:
		yao=f.readlines()
	yao16=pd.DataFrame(columns=['Word','Valence','Arousal'])
	yao16.set_index('Word', inplace=True)
	counter=0
	for row in yao:
		counter += 1
		if counter <= 18:
			continue
		if row.strip()=='':
			counter -= 1
			continue
		column=counter % 18
		if column==1:
			word=row.strip()
		if column==3:
			valence=row.strip()
		if column == 5:
			arousal=row.strip()
			yao16.loc[word]=[valence,arousal]
	return yao16
	#raise NotImplementedError

def load_monnier14():
	'''
	Monnier, C., & Syssau, A. (2014). Affective norms for french words (FAN). 
	Behavior Research Methods, 46(4), 1128–1137. 
	https://doi.org/10.3758/s13428-013-0431-1
	'''
	monnier14=pd.read_excel(cs.monnier14, skiprows=2, usecols=[0,4,6])
	monnier14.columns=heads_vad[:-1]
	monnier14.set_index('Word', inplace=True)
	return monnier14
	#raise NotImplementedError

def load_davidson14():
	'''
	Davidson, P., & Innes-Ker, Å. (2014). Valence and arousal norms for Swedish 
	affective words (Lund Psychological Reports No. Volume 14, No. 2). 
	Lund University.
	'''
	davidson14=pd.read_csv(cs.davidson14,sep='\t')
	davidson14=davidson14[['Ord','Valens','Arousal']]
	davidson14.columns=heads_vad[:-1]
	davidson14.set_index('Word', inplace=True)
	return davidson14
	#raise NotImplementedError

def load_eilola10():
	'''
	Eilola, T. M., & Havelka, J. (2010). Affective norms for 210 British 
	English and Finnish nouns. Behavior Research Methods, 42(1), 134–140.
	'''
	eilola10=pd.read_excel(cs.eilola10)
	eilola10=eilola10[['Finnish Word','Finnish Valence mean']]
	eilola10.columns=['Word','Valence']
	eilola10.set_index('Word', inplace=True)
	return eilola10
	#raise NotImplementedError

def load_engelthaler17():
	'''
	Engelthaler, T. & Hills, T.T. Behav Res (2017). https://doi.org/10.3758/s13428-017-0930-6
	'''
	engelthaler17=pd.read_csv(cs.engelthaler17)
	engelthaler17=engelthaler17[['word','mean']]
	engelthaler17.columns=['Word','Humor']
	engelthaler17.set_index('Word', inplace=True)
	return engelthaler17
	#raise NotImplementedError


###############################################################

####################  Bi-representational data sets  #########################
def get_english():
	return load_anew99().join(load_stevenson07(), how='inner')
	# print(english.head())
	# print(english.shape)

def get_spanish():
	return load_redondo07().join(load_ferre16(), how='inner')
	# print(spanish.head())
	# print(spanish.shape)

def get_polish():
	return load_imbir16().join(load_wierzba15(), how='inner')
	# print(polish.head())
	# print(polish.shape)

def get_german():
	angst=load_schmidtke14()
	angst.set_index(angst.index.str.lower(), inplace=True)
	return angst.join(load_briesemeister11(), how='inner')
	# print(german.head())
	# print(german.shape)





##################### Embeddings#############################################

def get_google_sgns(vocab_limit=None):
	return Embedding.from_word2vec_bin(
			path=cs.google_news_embeddings,
			vocab_limit=vocab_limit)

def get_facebook_fasttext_wikipedia(language, vocab_limit=None):
	return Embedding.from_fasttext_vec(
			path=cs.facebook_fasttext_wikipedia[language],
			vocab_limit=vocab_limit)

def get_facebook_fasttext_common_crawl(vocab_limit=None):
	return Embedding.from_fasttext_vec(
			path=cs.facebook_fasttext_common_crawl,
			vocab_limit=None,
			zipped=True,
			file='crawl-300d-2M.vec')

def get_sedoc17_embeddings(vocab_limit=None):
	return Embedding.from_raw_format(path=cs.sedoc17_embeddings)
