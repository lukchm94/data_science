'''
Code taken from: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
Further classification: https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
'''
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# engineer tracker data
def prepare_data_raw(df):
    le = LabelEncoder()
    df.columns = df.columns.str.replace(' ', '_')
    ind_var = df[['KPCi_ID', 'KPC_ID', 'Primary_Taxonomy', 'Control_owner_name', 'Control_owner_location',
                  'EY_reliant', 'SOX_Relevant', 'CFOA_Relevant', 'DE_test_result', 'DE_actual_date', 'DE_target_date']]
    ind_var = ind_var.fillna({'New_KPCi?': 'No'})
    ind_var = ind_var.fillna({'DE_test_result': 'DE green pass'})
    ind_var = ind_var.drop(ind_var[ind_var.DE_actual_date == 'SII'].index)
    ind_var = ind_var.drop(ind_var[ind_var.DE_actual_date == 'NETD'].index)
    rows = int(ind_var.shape[0])
    index = list()
    delayed = list()
    for n in range(rows):
        index.append(ind_var.iloc[n, 0])
        if int(ind_var.iloc[n, 9]) > int(ind_var.iloc[n, 10]):
            delayed.append(1)
        else:
            delayed.append(0)

    # encode strings to int
    col = len(ind_var.columns)
    for i in range(col - 2):
        ind_var.iloc[:, i] = le.fit_transform(ind_var.iloc[:, i])
    ind_var['Delayed'] = delayed
    var = ind_var[['KPCi_ID', 'KPC_ID', 'Primary_Taxonomy', 'Control_owner_name', 'Control_owner_location',
                   'EY_reliant', 'SOX_Relevant', 'CFOA_Relevant', 'DE_test_result', 'Delayed']]
    var = var.set_index('KPCi_ID')
    print(var.Delayed.value_counts())
    dataset = list()
    dataset = var.values.tolist()
    dataset = np.array(dataset)
    return dataset, index

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# load data
train = pd.read_csv(r'learn.csv')
test = pd.read_csv(r'test.csv')

train_dataset, train_index = prepare_data_raw(train)
test_dataset, test_index = prepare_data_raw(test)

# convert class column to integers
str_column_to_int(train_dataset, len(train_dataset[0]) - 1)
str_column_to_int(test_dataset, len(test_dataset[0]) - 1)
# normalize input variables
minmax = dataset_minmax(train_dataset)
normalize_dataset(train_dataset, minmax)
minmax_test = dataset_minmax(test_dataset)
normalize_dataset(test_dataset, minmax_test)

# split into input (X) and output (y) variables
x_train = train_dataset[:, 0:8]
y_train = train_dataset[:, 8]
x_test = test_dataset[:, 0:8]
y_test = test_dataset[:, 8]

# define the keras model
model = tf.keras.Sequential()
model.add(layers.Dense(12, input_dim=8, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(x_train, y_train, epochs=300, batch_size=15)

# make class predictions train
predictions = model.predict_classes(x_train)
# get kpci_id for results
train.columns = train.columns.str.replace(' ', '_')
test.columns = test.columns.str.replace(' ', '_')
fal_pos = 0
fal_neg = 0
tru_pos = 0
tru_neg = 0
errors = list()
tru_red = list()

for i in range(len(predictions) - 1):
    if predictions[i] == 1:
        if predictions[i] != y_train[i]:
            error = 'fal_neg', str(train_index[i]), str(predictions[i]), str(y_train[i])
            errors.append(error)
            fal_neg += 1
        else:
            tru_pos += 1
            delayed = str(train_index[i])
            tru_red.append(delayed)
    elif predictions[i] == 0:
        if predictions[i] != y_train[i]:
            error = 'fal_pos', str(train_index[i]), str(predictions[i]), str(y_train[i])
            errors.append(error)
            fal_pos += 1
        else:
            tru_neg += 1
print('\n--- TRAIN RESULTS ---')
print('\nTrue positive: ', tru_pos)
print('True negative: ', tru_neg)
print('False positive: ', fal_pos)
print('False negative: ', fal_neg)
accuracy = (tru_pos + tru_neg) / (tru_pos + tru_neg + fal_pos + fal_neg)
print('Accuracy: ' + '{:.2%}'.format(accuracy))

df_errors = pd.DataFrame(errors)
df_errors.to_csv('errors_train_2.csv', index=False, header=False)
df_tru_red = pd.DataFrame(tru_red)
df_tru_red.to_csv('predicted_red_train_2.csv', index=False, header=False)

# make class predictions test
predictions = model.predict_classes(x_test)
# get kpci_id for results
train.columns = train.columns.str.replace(' ', '_')
test.columns = test.columns.str.replace(' ', '_')
fal_pos = 0
fal_neg = 0
tru_pos = 0
tru_neg = 0
errors = list()
tru_red = list()

for i in range(len(predictions) - 1):
    if predictions[i] == 1:
        if predictions[i] != y_test[i]:
            error = 'fal_neg', str(test_index[i]), str(predictions[i]), str(y_test[i])
            errors.append(error)
            fal_neg += 1
        else:
            tru_pos += 1
            delayed = str(test_index[i])
            tru_red.append(delayed)
    elif predictions[i] == 0:
        if predictions[i] != y_test[i]:
            error = 'fal_pos', str(test_index[i]), str(predictions[i]), str(y_test[i])
            errors.append(error)
            fal_pos += 1
        else:
            tru_neg += 1
print('\n--- TEST RESULTS ---')
print('\nTrue positive: ', tru_pos)
print('True negative: ', tru_neg)
print('False positive: ', fal_pos)
print('False negative: ', fal_neg)
accuracy = (tru_pos + tru_neg) / (tru_pos + tru_neg + fal_pos + fal_neg)
print('Accuracy: ' + '{:.2%}'.format(accuracy))

df_errors = pd.DataFrame(errors)
df_errors.to_csv('errors_test_2.csv', index=False, header=False)
df_tru_red = pd.DataFrame(tru_red)
df_tru_red.to_csv('predicted_red_test_2.csv', index=False, header=False)

'''
# evaluate the keras model
_, accuracy = model.evaluate(x_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))
'''