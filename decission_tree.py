import pandas as pd
import datetime as dt
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
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

def dec_tree(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
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
    '''
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=X_train.columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('test.png')
    Image(graph.create_png())
    '''

def log_reg(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print('\n', cm, '\n')
    print('\nCLASSIFICATION REPORT - LOG REG\n')
    print(classification_report(y_test, y_pred))

def check_for_nan(df):
    print(df.isnull().any())
    print(df.shape)

    for col in df.columns:
        if df[col].isnull().any() == True:
            print('Count of number of NaN in ', col, ' column: ', df[col].isnull().sum())
            print(df[col].value_counts())

df = pd.read_csv(r'in-vehicle-coupon-recommendation.csv')
print(df.Y.value_counts())
check_for_nan(df)
df = df.drop('car', 1)
df = df.drop('toCoupon_GEQ5min', 1)
df = df.fillna({'Bar': 'never'})
df = df.fillna({'CoffeeHouse': 'less1'})
df = df.fillna({'CarryAway': '1~3'})
df = df.fillna({'RestaurantLessThan20': '1~3'})
df = df.fillna({'Restaurant20To50': '1~3'})
check_for_nan(df)

target_column = ['Y']
predictors = list(set(list(df.columns)) - set(target_column))
x = df[predictors]
y = df[target_column]

col = len(x.columns)
for i in range(col):
    x.iloc[:, i] = le.fit_transform(x.iloc[:, i])

x_scaled = scaling(x)
cm = x_scaled.corr()
sn.heatmap(cm, annot=False, linewidths=0.5, cmap='ocean')
plt.show()
dec_tree(x, y)
log_reg(x, y)
dec_tree(x_scaled, y)
log_reg(x_scaled, y)
'''
df_gf = pd.read_csv(r'test.csv')
df_gops = pd.read_csv(r'tracker.csv')
x, y = prepare_data(df)
x_scaled = scaling(x)
x_gf, y_gf = prepare_data_raw(df_gf)
x_gf_scaled = scaling(x_gf)
x_gops, y_gops = prepare_data(df_gops)
x_gops_scaled = scaling(x_gops)

'''