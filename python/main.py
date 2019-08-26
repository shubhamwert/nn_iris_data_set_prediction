import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_iris
import scipy as sc
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data=load_iris()
x=sc.asfarray(data.data)

#checking x for null values
print(sum(sc.isnan(x)))

y=sc.asfarray(data.target)
#checking y for null values
print(sum(sc.isnan(y)))

#labels and there mappings
labels=data.target_names
labelDict={
0:labels[0],
1:labels[1],
2:labels[2],

}

#NO visualization is done in this practice

#spilliting data

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)
print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(20, activation=tf.nn.relu),
  tf.keras.layers.Dense(3,activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=100)
predictions = model.predict(x_test)


