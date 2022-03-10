import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

digit=2
data_type = 'Emp'

#Clean data
msa_pop = pd.read_excel(f'data/Hyejin_raw_data/msa.xlsx')
ind_name = pd.read_excel(f'data/Hyejin_raw_data/ind{digit}.xlsx')


df_list = []
for i in range(16):
    data = pd.read_excel(f'data/Hyejin_raw_data/size{digit}{data_type}.xlsx',sheet_name=i,header=None).transpose()
    data['Population'] = msa_pop[f'pop{1998+i}']
    data['area'] = msa_pop['area']
    data['area_code'] = msa_pop['code']
    data['Time'] = i

    data = data[ data.sum(axis=1)>0 ]
    df_list.append(data)

df = pd.concat(df_list)
df = df[~df['Population'].isna()].reset_index(drop=True)
for col in range(len(ind_name)):
    df.rename(columns={col:ind_name['industry'][col]})
# df.to_csv('data/Hyejin_processed_data.csv')

df_ratio = df.iloc[:,:19].div(df['Population'],axis=0)
ind_ratio_all_samples = df.iloc[:,:19].sum(axis=0)/df['Population'].sum()
df_rca = df_ratio/ind_ratio_all_samples
df_rca.insert(loc=0, column='Label', value=df['area'])
df_rca.insert(loc=1, column='Time', value=df['Time'])

df_rca.to_csv(f'data/hyejin2020_size{digit}{data_type}_rca.csv',index=False)

#Try PCA on ratio
pca_model_ratio = PCA()
pca_model_ratio.fit(df_ratio)
print('explained variance ratio',pca_model_ratio.explained_variance_ratio_)
pca_ratio_value = pca_model_ratio.transform(df_ratio)

#Try PCA on RCA
pca_model_rca = PCA()
pca_model_rca.fit(df_rca)
print('explained variance ratio',pca_model_rca.explained_variance_ratio_)
pca_rca_value = pca_model_rca.transform(df_rca)
