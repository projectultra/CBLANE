from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.metrics import AUC, BinaryAccuracy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import os
from keras.callbacks import Callback
from keras.models import save_model,load_model
from dataset import save_or_load_numpy

CBLANE = load_model("CBLANE\models\CBLANE_global.keras")

train_labels,train_features = save_or_load_numpy("load","train.npz")
test_labels,test_features = save_or_load_numpy("load","test.npz")
validation_labels,validation_features = save_or_load_numpy("load","validation.npz")

reduce_lr = ReduceLROnPlateau(monitor='val_loss',mode="min", factor=0.1, patience=1, min_lr=1e-8)
early_stop = EarlyStopping(monitor='val_loss',mode="min", patience=10, restore_best_weights=True)

class SaveSubModels(Callback):
    def on_epoch_end(self, epoch, logs=None):
        model_directories = [f'model/{epoch}/CBLANE.keras']

        for directory in set(os.path.dirname(model_path) for model_path in model_directories):
            if not os.path.exists(directory):
                os.makedirs(directory)

        save_model(CBLANE, model_directories[0])

CBLANE.compile(loss='binary_crossentropy',
                             optimizer=Adam(learning_rate=0.001),
                             metrics=[BinaryAccuracy(),
                                      AUC(),
                                      ]
                             )

history = CBLANE.fit(tf.constant(train_features,dtype=tf.bool),
                                   tf.constant(train_labels,dtype=tf.bool),
                                   batch_size=128,
                                   epochs=5,
                                   verbose=1,
                                   validation_data=(tf.constant(validation_features,dtype = tf.bool),
                                                    tf.constant(validation_labels,dtype = tf.bool)),
                                   callbacks=([
                                       SaveSubModels
                                               ]),
                                  validation_batch_size=4096)

CBLANE.save_model("CBLANE_global_dataset.keras")

small_train_features = []
small_train_labels = []
small_test_features = []
small_test_labels = []
for trainset,testset in np.load("small_dataset.npz",allow_pickle=True)['arr_0']:
  small_train_features.append(np.unpackbits(trainset[0],axis=2,count=4))
  small_train_labels.append(np.unpackbits(trainset[1],axis=-1,count=len(small_train_features[-1])))
  small_test_features.append(np.unpackbits(testset[0],axis=2,count=4))
  small_test_labels.append(np.unpackbits(testset[1],axis=-1,count=len(small_test_features[-1])))

CBLANE.compile(loss='binary_crossentropy',
                             optimizer=Adam(learning_rate=0.001),
                             metrics=[BinaryAccuracy(),
                                      AUC()
                                      ]
                             )
history = CBLANE.fit(tf.constant(small_train_features,dtype=tf.bool),
                             tf.constant(small_train_labels,dtype=tf.bool),
                             batch_size=1024,
                             epochs=30,
                             verbose=1,
                             validation_split=0.1,
                             validation_batch_size=4096,
                             callbacks=([reduce_lr]))

CBLANE.save_model("CBLANE_small_dataset.keras")

i = 0
avg_acc = []
for train_feature,train_label,test_feature,test_label in zip(small_train_features,small_train_labels,small_test_features,small_test_labels):
  i=i+1
  print("Dataset:",i)
  CBLANE = load_model("CBLANE_small_dataset.keras")
  CBLANE.compile(loss='binary_crossentropy',
                              optimizer=Adam(learning_rate=0.0001),
                              metrics=[BinaryAccuracy(),
                                        AUC(),
                                        ]
                              )
  history = CBLANE.fit(tf.constant(train_feature,dtype=tf.bool),
                              tf.constant(train_label,dtype=tf.bool),
                              batch_size=1024,
                              epochs=30,
                              verbose=0,
                              validation_split=0.1,
                              validation_batch_size=4096,
                              callbacks=([reduce_lr,
                                          early_stop]))
  data = CBLANE.evaluate(test_feature,test_label,verbose=1)
  avg_acc.append(data[2])

medium_train_features = []
medium_train_labels = []
medium_test_features = []
medium_test_labels = []
for trainset,testset in np.load("medium_dataset.npz",allow_pickle=True)['arr_0']:
  medium_train_features.append(np.unpackbits(trainset[0],axis=2,count=4))
  medium_train_labels.append(np.unpackbits(trainset[1],axis=-1,count=len(medium_train_features[-1])))
  medium_test_features.append(np.unpackbits(testset[0],axis=2,count=4))
  medium_test_labels.append(np.unpackbits(testset[1],axis=-1,count=len(medium_test_features[-1])))

CBLANE.compile(loss='binary_crossentropy',
                             optimizer=Adam(learning_rate=0.0001),
                             metrics=[BinaryAccuracy(),
                                      AUC()
                                      ]
                             )
history = CBLANE.fit(tf.constant(medium_train_features,dtype=tf.bool),
                             tf.constant(medium_train_labels,dtype=tf.bool),
                             batch_size=1024,
                             epochs=30,
                             verbose=1,
                             validation_split=0.1,
                             validation_batch_size=4096,
                             callbacks=([reduce_lr]))

CBLANE.save_model("CBLANE_medium_dataset.keras")

i = 0
avg_acc = []
for train_feature,train_label,test_feature,test_label in zip(medium_train_features,medium_train_labels,medium_test_features,medium_test_labels):
  i=i+1
  print("Dataset:",i)
  CBLANE = load_model("CBLANE_medium_dataset.keras")
  CBLANE.compile(loss='binary_crossentropy',
                              optimizer=Adam(learning_rate=0.0001),
                              metrics=[BinaryAccuracy(),
                                        AUC(),
                                        ]
                              )
  history = CBLANE.fit(tf.constant(train_feature,dtype=tf.bool),
                              tf.constant(train_label,dtype=tf.bool),
                              batch_size=1024,
                              epochs=30,
                              verbose=0,
                              validation_split=0.1,
                              validation_batch_size=4096,
                              callbacks=([reduce_lr,
                                          early_stop]))
  data = CBLANE.evaluate(test_feature,test_label,verbose=1)
  avg_acc.append(data[2])

large_train_features = []
large_train_labels = []
large_test_features = []
large_test_labels = []
for trainset,testset in np.load("large_dataset.npz",allow_pickle=True)['arr_0']:
  large_train_features.append(np.unpackbits(trainset[0],axis=2,count=4))
  large_train_labels.append(np.unpackbits(trainset[1],axis=-1,count=len(large_train_features[-1])))
  large_test_features.append(np.unpackbits(testset[0],axis=2,count=4))
  large_test_labels.append(np.unpackbits(testset[1],axis=-1,count=len(large_test_features[-1])))

CBLANE.compile(loss='binary_crossentropy',
                             optimizer=Adam(learning_rate=0.0001),
                             metrics=[BinaryAccuracy(),
                                      AUC()]
                             )
history = CBLANE.fit(tf.constant(large_train_features,dtype=tf.bool),
                             tf.constant(large_train_labels,dtype=tf.bool),
                             batch_size=1024,
                             epochs=30,
                             verbose=1,
                             validation_split=0.1,
                             validation_batch_size=4096,
                             callbacks=([reduce_lr]))

CBLANE.save_model("CBLANE_large_dataset.keras")

i = 0
avg_acc = []
for train_feature,train_label,test_feature,test_label in zip(large_train_features,large_train_labels,large_test_features,large_test_labels):
  i=i+1
  print("Dataset:",i)
  CBLANE = load_model("CBLANE_large_dataset.keras")
  CBLANE.compile(loss='binary_crossentropy',
                              optimizer=Adam(learning_rate=0.001),
                              metrics=[BinaryAccuracy(),
                                        AUC(),
                                        ]
                              )
  history = CBLANE.fit(tf.constant(train_feature,dtype=tf.bool),
                              tf.constant(train_label,dtype=tf.bool),
                              batch_size=1024,
                              epochs=30,
                              verbose=0,
                              validation_split=0.1,
                              validation_batch_size=4096,
                              callbacks=([reduce_lr,
                                          early_stop]))
  data = CBLANE.evaluate(test_feature,test_label,verbose=1)
  avg_acc.append(data[2])