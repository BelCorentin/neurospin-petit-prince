import pandas as pd


df = pd.read_csv('annotation FR lppFR_word_information.csv')
df_ = pd.DataFrame(df['onset'])

df_['duration'] = df['offset'] - df['onset']

df_['trial_type'] = [{"kind":"word",'word':df.loc[i,'word'].replace("'","")} for i in range(df_.shape[0])]


print(df_)

df_.to_csv('./annotation_processed.tsv',sep='\t',index=False)