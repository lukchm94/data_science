import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import tkinter as tk
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

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
    for col in df.columns:
        if df[col].isnull().any() == True:
            print('Count of number of NaN in ', col, ' column: ', df[col].isnull().sum())
            print(df[col].value_counts())
            print(df[col].mode()[0])
            if df[col].isnull().sum() <= (df.shape[0]/2):
                df[col].fillna(str(df[col].mode()[0]), inplace=True)
            else:
                del df[col]
    return df

def maximum_absolute_scaling(df):
    df_scaled = df.copy()
    for column in df_scaled.columns:
        df_scaled[column] = df_scaled[column]  / df_scaled[column].abs().max()
    return df_scaled

def encode_int(x):
    le = LabelEncoder()
    col = len(x.columns)
    for i in range(col):
        x.iloc[:, i] = le.fit_transform(x.iloc[:, i])
    return x

def log_reg_complete(df, abs_scaler):
    x_scaled, y_scaled, x_int, y_int = prepare_classification(df, abs_scaler)
    cm = x_scaled.corr()
    sn.heatmap(cm, annot=False, linewidths=0.5, cmap='ocean')
    plt.show()
    print('\n----- ORIGINAL RESULTS -----\n')
    log_reg(x_int, y_int)
    print('\n----- SCALED RESULTS -----\n')
    log_reg(x_scaled, y_scaled)

def dec_tree_complete(df, abs_scaler):
    x_scaled, y_scaled, x_int, y_int = prepare_classification(df, abs_scaler)
    cm = x_scaled.corr()
    sn.heatmap(cm, annot=False, linewidths=0.5, cmap='ocean')
    plt.show()
    print('\n----- ORIGINAL RESULTS -----\n')
    dec_tree(x_int, y_int)
    print('\n----- SCALED RESULTS -----\n')
    dec_tree(x_scaled, y_scaled)

def mlp_network(df, abs_scaler):
    df_train, df_test = train_test_split(df, test_size=0.33)
    x_scaled_train, y_scaled_train, x_int_train, y_int_train = prepare_classification(df_train, abs_scaler)
    x_scaled_test, y_scaled_test, x_int_test, y_int_test = prepare_classification(df_train, abs_scaler)
    cm = x_scaled_train.corr()
    sn.heatmap(cm, annot=False, linewidths=0.5, cmap='ocean')
    plt.show()
    # define the keras model
    model = tf.keras.Sequential()
    model.add(layers.Dense(25, input_dim=24, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(x_scaled_train, y_scaled_train, epochs=300, batch_size=15)
    # make class predictions train
    predictions = model.predict_classes(x_scaled_test)
    # evaluate the keras model
    _, accuracy = model.evaluate(x_scaled_test, y_scaled_test)
    print('Accuracy: %.2f' % (accuracy * 100))

def prepare_classification(df, abs_scaler):
    print(df.Y.value_counts())
    df_checked = check_for_nan(df)
    df_int = encode_int(df_checked)
    abs_scaler.fit(df_int)
    abs_scaler.max_abs_
    scaled_data = abs_scaler.transform(df_int)
    df_scaled = pd.DataFrame(scaled_data, columns=df_int.columns)
    target_column = ['Y']
    predictors_scaled = list(set(list(df_scaled.columns)) - set(target_column))
    predictors_int = list(set(list(df_int.columns)) - set(target_column))
    x_scaled = df_scaled[predictors_scaled]
    y_scaled = df_scaled[target_column]
    x_int = df_int[predictors_int]
    y_int = df_int[target_column]
    return x_scaled, y_scaled, x_int, y_int

#setting variables
abs_scaler = MaxAbsScaler()
df = pd.read_csv(r'in-vehicle-coupon-recommendation.csv')
#Main loop for the code
window = tk.Tk()
tree = tk.Button(text='Run Decision Tree', width=30, height=10, command=lambda: dec_tree_complete(df, abs_scaler))
reg = tk.Button(text='Run Logistic Regression', width=30, height=10, command=lambda: log_reg_complete(df, abs_scaler))
mlp = tk.Button(text='Run Neural Network Algorithm', width=30, height=10, command=lambda: mlp_network(df, abs_scaler))
tree.pack()
reg.pack()
mlp.pack()
window.mainloop()