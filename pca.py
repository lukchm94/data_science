import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wines = load_wine()
df = pd.DataFrame(wines.data, columns=wines.feature_names)
print(df.head(5))

print(df.shape, '\n', df.columns)

print(df.dtypes)
print(df.iloc[:,0:4].describe())

df_scaled = StandardScaler().fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=wines.feature_names)

print('\nAfter scaling\n',df_scaled.iloc[:,0:4].describe())
print('\nApplying PCA analysis\n')

pca = PCA(n_components=2)
pca_model = pca.fit(df_scaled)
df_trans = pd.DataFrame(pca_model.transform(df_scaled), columns=['pca1','pca2'])

print(df_trans.describe())

plt.scatter(df_trans['pca1'], df_trans['pca2'], alpha=0.8)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

plt.show()



