import pandas as pd
import datetime as dt
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

now = str(dt.datetime.now())
now = now.replace(' ', '_')
le = LabelEncoder()
sc = MinMaxScaler()

def scaling(df):
    df_scaled = df.copy()
    for column in df_scaled.columns:
        df_scaled[column] = df_scaled[column] / df_scaled[column].abs().max()
    return df_scaled

def prepare_data(df):
    df.columns = df.columns.str.replace(' ', '_')
    ind_var = df[['KPCi_ID', 'DE_test_result', 'Primary_Taxonomy',
                  'Control_owner_name', 'Control_owner_location',
                  'EY_reliant', 'SOX_Relevant', 'CFOA_Relevant']]
    ind_var = ind_var.fillna({'New_KPCi?': 'No'})
    ind_var = ind_var.fillna({'DE_test_result': 'DE green pass'})
    col = len(ind_var.columns)
    for i in range(col):
        if ind_var.columns[i] != 'KPCi_ID':
            if ind_var.columns[i] != 'DE_test_result':
                ind_var.iloc[:, i] = le.fit_transform(ind_var.iloc[:, i])

    ind_var = ind_var.set_index('KPCi_ID')
    ind_var = ind_var.drop(ind_var[ind_var.DE_test_result == 'Not applicable'].index)
    corrMatrix = ind_var.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()

    rows = int(ind_var.shape[0])
    for n in range(rows):
        if ind_var.iloc[n, 0] == 'DE red fail':
            for i in range(10):
                ind_var = ind_var.append(ind_var.iloc[n, :], ignore_index=True)
        elif ind_var.iloc[n, 0] == 'SII':
            for j in range(10):
                ind_var = ind_var.append(ind_var.iloc[n, :], ignore_index=True)
        elif ind_var.iloc[n, 0] == 'NETD':
            for k in range(10):
                ind_var = ind_var.append(ind_var.iloc[n, :], ignore_index=True)
        elif ind_var.iloc[n, 0] == 'DE amber pass':
            for l in range(10):
                ind_var = ind_var.append(ind_var.iloc[n, :], ignore_index=True)

    target_column = ['DE_test_result']
    predictors = list(set(list(ind_var.columns)) - set(target_column))
    x = ind_var[predictors]
    y = ind_var[target_column]
    return x, y

def prepare_data_raw(df):
    df.columns = df.columns.str.replace(' ', '_')
    ind_var = df[['KPCi_ID', 'DE_test_result', 'Primary_Taxonomy',
                  'Control_owner_name', 'Control_owner_location',
                  'EY_reliant', 'SOX_Relevant', 'CFOA_Relevant']]
    ind_var = ind_var.fillna({'New_KPCi?': 'No'})
    ind_var = ind_var.fillna({'DE_test_result': 'DE green pass'})
    col = len(ind_var.columns)
    for i in range(col):
        if ind_var.columns[i] != 'KPCi_ID':
            if ind_var.columns[i] != 'DE_test_result':
                ind_var.iloc[:, i] = le.fit_transform(ind_var.iloc[:, i])

    ind_var = ind_var.set_index('KPCi_ID')
    ind_var = ind_var.drop(ind_var[ind_var.DE_test_result == 'Not applicable'].index)
    target_column = ['DE_test_result']
    predictors = list(set(list(ind_var.columns)) - set(target_column))
    x = ind_var[predictors]
    y = ind_var[target_column]
    return x, y

def dec_tree(x, y, x_gf, y_gf):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print('\nTRAIN SHAPE - DECISSION TREE\n')
    print(X_train.shape, y_train.shape)
    print('\nTEST SHAPE - DECISSION TREE\n')
    print(X_test.shape, y_test.shape)
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print('\n', cm, '\n')
    print('\nCLASSIFICATION REPORT - DECISSION TREE\n')
    print(classification_report(y_test, y_pred))
    df_results = pd.DataFrame(y_pred)
    df_test = pd.DataFrame(y_test)
    index = X_test.index.tolist()
    df_results.insert(0, 'KPCi_ID', index)
    df_results = df_results.set_index('KPCi_ID')
    y_pred_fin = clf.predict(x_gf)
    index_fin = x_gf.index.tolist()
    results_fin = pd.DataFrame({'Model_results': y_pred_fin}, index=index_fin)
    fin_actuals = pd.DataFrame(y_gf)
    result = pd.concat([results_fin, fin_actuals], axis=1)
    result.to_csv(r'dt.csv',
                  index=True, header=True)
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=X_train.columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('test.png')
    Image(graph.create_png())
    cm = confusion_matrix(y_gf, y_pred_fin)
    print('\nCONFUSION MATRIX ACTUALS FIN\n', cm, '\n')
    print('\nCLASSIFICATION REPORT ACTUALS FIN - DECISSION TREE\n')
    print(classification_report(y_gf, y_pred_fin))

df = pd.read_csv(r'learn.csv')
df_gf = pd.read_csv(r'test.csv')
df_gops = pd.read_csv(r'tracker.csv')
x, y = prepare_data(df)
x_scaled = scaling(x)
x_gf, y_gf = prepare_data_raw(df_gf)
x_gf_scaled = scaling(x_gf)
x_gops, y_gops = prepare_data(df_gops)
x_gops_scaled = scaling(x_gops)
dec_tree(x_scaled, y, x_gf_scaled, y_gf)