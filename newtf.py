# import tensorflow as tf
# import numpy as np

# c = np.array([[3.,4], [5.,6], [6.,7]])
# step = tf.reduce_mean(c, 1)                                                                                 
# with tf.Session() as sess:
#     print(sess.run(step))




import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd

list1 = ['Past_10_h', 'Past_10_v']
'Past_10_h', 'Past_10_v', 'h_ML', 'rating1_pre', 'rating2_pre'
train_years = ['2011', '2013', '2014', '2015', '2016', '2017']
test_years = ['2018']
sport = 'mlb'
train_fns = ['./data/' + year + sport + '.csv' for year in train_years]
test_fns = ['./data/' + year + sport + '.csv' for year in test_years]
def clean(fn):
    df = pd.read_csv(fn)
    df = df.dropna()
    df = df.drop_duplicates()
    return df  
def dfs(fns):
    dfs = []
    for fn in fns:
        tmp_df = clean(fn)
        dfs.append(tmp_df)
    df = pd.concat(dfs)
    return df 


train_df = dfs(train_fns)
test_df = dfs(test_fns)

feature_cols = list1
x = train_df[feature_cols]
y = train_df['h_win']

x = x.values
y = y.values

model = tf.keras.Sequential([
    #keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x, y, epochs=450, batch_size=32,)
test_x = test_df[feature_cols]
test_y = test_df['h_win']

test_x = test_x.values
test_y = test_y.values
test_loss, test_acc = model.evaluate(test_x, test_y)

print('Test accuracy:', test_acc)
test = [[.6, .8]]
test = np.asarray(test)
prediction = model.predict(test)
#prediction = np.argmax(prediction)
print(prediction)
prediction = np.argmax(prediction)
print(prediction)


