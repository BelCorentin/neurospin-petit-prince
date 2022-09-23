import pandas as pd


df = pd.read_csv('./annotation_processed.tsv',sep='\t')

df1 = df.iloc[0:1632,:]
df1.to_csv('./annotation_processed1.tsv',sep='\t',index=False)

print(df1)
df2 = df.iloc[1632:3419,:]
df2.to_csv('./annotation_processed2.tsv',sep='\t',index=False)

print(df2)
df3 = df.iloc[3419:5295,:]
df3.to_csv('./annotation_processed3.tsv',sep='\t',index=False)

print(df3)
df4 = df.iloc[5295:6945,:]
df4.to_csv('./annotation_processed4.tsv',sep='\t',index=False)

print(df4)
df5 = df.iloc[6945:8472,:]
df5.to_csv('./annotation_processed5.tsv',sep='\t',index=False)

print(df5)
df6 = df.iloc[8472:10330,:]
df6.to_csv('./annotation_processed6.tsv',sep='\t',index=False)

print(df6)
df7 = df.iloc[10330:12042,:]
df7.to_csv('./annotation_processed7.tsv',sep='\t',index=False)

print(df7)
df8 = df.iloc[12042:13581,:]
df8.to_csv('./annotation_processed8.tsv',sep='\t',index=False)

print(df8)
df9 = df.iloc[13581:15391,:]
df9.to_csv('./annotation_processed9.tsv',sep='\t',index=False)
print(df9)

