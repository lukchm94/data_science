import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

now = str(dt.datetime.now())
df = pd.read_csv(r'tracker_3_2_2021.csv')

print(df.iloc[:, 35].value_counts())
rows = int(df.shape[0])
print(rows)

for n in range(rows):
    if df.iloc[n, 35] == 'DE red pass':
        for i in range(100):
            df = df.append(df.iloc[n, :], ignore_index=True)
    elif df.iloc[n, 35] == 'DE amber pass':
        for j in range(100):
            df = df.append(df.iloc[n, :], ignore_index=True)
print(df.iloc[:, 35].value_counts())
ind_var = df[['KPCi ID', 'DE test result', 'Primary Taxonomy', 'Workstream',
              'Control owner name', 'Control owner location', 'Frequency', 'New KPCi?',
              'EY reliant', 'SOX Relevant', 'CFOA Relevant', 'BCBS relevant']]
ind_var = ind_var.fillna({'New KPCi?': 'No'})
ind_var = pd.get_dummies(ind_var, columns=['Primary Taxonomy'], prefix=['PT'])
ind_var = pd.get_dummies(ind_var, columns=['Workstream'], prefix=['WS'])
ind_var = pd.get_dummies(ind_var, columns=['Control owner name'], prefix=['CO_name'])
ind_var = pd.get_dummies(ind_var, columns=['Control owner location'], prefix=['CO_location'])
ind_var = pd.get_dummies(ind_var, columns=['Frequency'], prefix=['freq'])
ind_var = pd.get_dummies(ind_var, columns=['EY reliant'], prefix=['EY'])
ind_var = pd.get_dummies(ind_var, columns=['SOX Relevant'], prefix=['SOX'])
ind_var = pd.get_dummies(ind_var, columns=['CFOA Relevant'], prefix=['CFOA'])
ind_var = pd.get_dummies(ind_var, columns=['BCBS relevant'], prefix=['BCBS'])
ind_var = pd.get_dummies(ind_var, columns=['New KPCi?'], prefix=['new_kpci'])
ind_var = ind_var.set_index('KPCi ID')
target_column = ['DE test result']
predictors = list(set(list(ind_var.columns)) - set(target_column))
xVar = ind_var[predictors]
yVar = ind_var[target_column]
X_train, X_test, y_train, y_test = train_test_split(xVar, yVar, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# Building and training the model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Generating accuracy, precision, recall and f1-score
target_names = ['DE green pass', 'DE amber pass', 'DE red pass']
print(classification_report(y_test, y_pred))
df_results = pd.DataFrame(y_pred)
df_test = pd.DataFrame(y_test)
print(df_test.head())
print(df_results.head())
df_results.to_csv(r'df_results_hd.csv', index=True, header=True)
df_test.to_csv(r'df_test_hd.csv', index=True, header=True)