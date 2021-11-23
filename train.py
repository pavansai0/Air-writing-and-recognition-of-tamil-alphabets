import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense,Activation,Flatten,Conv2D,MaxPooling2D,Dropout
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
import numpy as np
import pickle
X,Y=pickle.load(open('tam.pickle','rb'))
X=np.array(X).reshape(-1,100,100,1)
Y=np.array(Y).reshape(-1,1)
def learn():
    model=tf.keras.models.Sequential()
    model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))


    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(18))
    model.add(Activation('softmax'))

    model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    monitor=EarlyStopping(monitor='val_accuracy',patience=100,restore_best_weights=True)
    model.fit(X,Y,batch_size=16,validation_split=0.1,epochs=200,verbose=2,callbacks=[monitor])
    model.save('fin.h5')
learn()
