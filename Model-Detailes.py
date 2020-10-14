
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# pip install python_speech_features
import numpy as np
import csv
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from python_speech_features import mfcc, delta, logfbank
from random import randint
import librosa


FileNames = {"train":"ml-fmi-23-2020//train.txt", "valid":"ml-fmi-23-2020//validation.txt", "test":"ml-fmi-23-2020//test.txt", "ex":"ml-fmi-23-2020//sample_submission.txt","pred":"ml-fmi-23-2020//predictions.txt"}

AudioFolders = {"train":"ml-fmi-23-2020//audio//train//", "valid":"ml-fmi-23-2020//audio//validation//", "test":"ml-fmi-23-2020//audio//test//"}

sr   = 16000 # Sample Rate   - 16 kHz
wlen = 0.025 # window length - 25 ms = 400 samples
slen = 0.01  # step   length - 10 ms = 160 samples 
nfft = 512 

def readCsv (fileName, hasLables):
    data = []
    with open(fileName, "r", newline='\n') as csvfile:
        for row in csv.reader(csvfile, delimiter=','):
            data.extend(row)
    if hasLables:
        data = np.transpose(np.array(data).reshape((len(data)//2, 2))) 
    else:
        data = np.array(data)
    return data

def writeCsv (data, labels, fileName = FileNames['pred']):
    with open(fileName, "w", newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['name', 'label'])
        for row in np.transpose([data, labels]):
            writer.writerow(row)

def getData (folderName, dataNames):
    data = []
    for dataName in dataNames:
        audio = librosa.load(AudioFolders[folderName]+dataName)[0]
        data.append(librosa.amplitude_to_db(abs(librosa.stft(audio))))#.reshape(-1))
        # data.append(np.abs(fft(wav.read(AudioFolders[folderName]+dataName)[1])))
        # data.append(mfcc(wav.read(AudioFolders[folderName]+dataName)[1], numcep=12))#.reshape(-1))
        
    return np.array(data)


# %%



# %%
train_data_name, train_labels = readCsv(FileNames['train'], hasLables=True)
valid_data_name, valid_labels = readCsv(FileNames['valid'], hasLables=True)
test_data_name                = readCsv(FileNames['test'],  hasLables=False)
# writeCsv(train_data, train_labels)


# %%
# train_data = getData('train', train_data_name)
# valid_data = getData('valid', valid_data_name)
# test_data  = getData('test',  test_data_name)
# mi = min([train_data.min(), valid_data.min(), test_data.min()])
# ma = max([train_data.max(), valid_data.max(), test_data.max()])
# print (mi, ma)
# train = (train_data - mi) / (ma-mi)
# valid = (valid_data - mi) / (ma-mi)
# test =  (test_data  - mi) / (ma-mi)
# mi = min([train.min(), valid.min(), test.min()])
# ma = max([train.max(), valid.max(), test.max()])
# print (mi, ma)
# train.shape, valid.shape, test.shape


# %%
initialShape = train.shape
train = train.reshape(initialShape[0], initialShape[1], initialShape[2], 1)
initialShape = valid.shape
valid = valid.reshape(initialShape[0], initialShape[1], initialShape[2], 1)
initialShape = test.shape
test = test.reshape(initialShape[0], initialShape[1], initialShape[2], 1)
train.shape, valid.shape, test.shape


# %%
# train_data1 = getData('train', train_data_name)
# valid_data1 = getData('valid', valid_data_name)
# test_data1 = getData('test',  test_data_name)
# mi = min([train_data1.min(), valid_data1.min(), test_data1.min()])
# ma = max([train_data1.max(), valid_data1.max(), test_data1.max()])
# print (mi, ma)
# train1 = (train_data1 - mi) / (ma-mi)
# valid1 = (valid_data1 - mi) / (ma-mi)
# test1 =  (test_data1  - mi) / (ma-mi)
# mi = min([train1.min(), valid1.min(), test1.min()])
# ma = max([train1.max(), valid1.max(), test1.max()])
# print (mi, ma)
# train1.shape, valid1.shape, test1.shape


# %%
initialShape = train1.shape
train1 = train1.reshape(initialShape[0], initialShape[1], initialShape[2], 1)
initialShape = valid1.shape
valid1 = valid1.reshape(initialShape[0], initialShape[1], initialShape[2], 1)
initialShape = test.shape
test1 = test1.reshape(initialShape[0], initialShape[1], initialShape[2], 1)
train1.shape, valid1.shape, test1.shape


# %%
train, train1 = train1, train
valid, valid1 = valid1, valid
test,  test1  = test1,  test
train.shape, valid.shape, test.shape


# %%



# %%
# 1
noModel = '1-stft'
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
model = Sequential()

model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1), padding='same', input_shape = train[0].shape))
model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1), padding='same'))
model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1), padding='same'))
model.add(MaxPool2D((3,3)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])#, 'mae', 'mse'])

model.summary()


# %%
for epoch in range(1,15):
    model.fit(
        x = train, 
        y = np.array([[1,0] if label == '0' else [0,1] for label in train_labels]), 
        epochs = epoch+1, 
        initial_epoch = epoch,
        verbose = 1, # progress bar
        # validation_split = 0.1,
        validation_data = (valid, np.array([[1,0] if label == '0' else [0,1] for label in valid_labels])),
        shuffle = True
    )
    pred = np.array(['0' if a > b else '1' for a, b in model.predict(valid)])
    model.save('ml-fmi-23-2020//Models//' + noModel + '//' + str(epoch) + '-' + str(sum(pred==valid_labels)))


# %%
epoch = '4-753' # epoch-validScore
saved_model = load_model('ml-fmi-23-2020//Models//' + noModel + '//' + epoch)
pred = np.array(['0' if a > b else '1' for a, b in saved_model.predict(valid)])
sum(pred==valid_labels)


# %%
pred = np.array(['0' if a > b else '1' for a, b in saved_model.predict(test)])
writeCsv(test_data_name, pred, 'ml-fmi-23-2020//Models//'+ model +'//predictions-' + model + '_' + epoch + '.txt')


# %%
# 2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
model = Sequential()

model.add(Conv2D(16, (5,4), activation='relu', strides=1, padding='same', input_shape = train[0].shape))
model.add(Conv2D(16, (5,4), activation='relu', strides=1, padding='same'))
model.add(MaxPool2D((5,4)))

model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# %%
for epoch in range(15):
    model.fit(
        x = train, 
        y = np.array([[1,0] if label == '0' else [0,1] for label in train_labels]), 
        epochs = 1, 
        verbose = 1, # progress bar
        # validation_split = 0.1,
        validation_data = (valid, np.array([[1,0] if label == '0' else [0,1] for label in valid_labels])),
        shuffle = True
    )
    pred = np.array(['0' if a > b else '1' for a, b in model.predict(valid)])
    model.save('ml-fmi-23-2020//Models//2//' + str(epoch) + '-' + str(sum(pred==valid_labels)))


# %%
# 3
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
model = Sequential()

model.add(Conv2D(16, (5,4), activation='relu', strides=1, padding='same', input_shape = train[0].shape))
model.add(Conv2D(16, (5,4), activation='relu', strides=1, padding='same'))
model.add(MaxPool2D((5,4)))
model.add(Conv2D(32, (5,4), activation='relu', strides=1, padding='same'))
model.add(MaxPool2D((5,4)))

model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# %%
for epoch in range(15):
    model.fit(
        x = train, 
        y = np.array([[1,0] if label == '0' else [0,1] for label in train_labels]), 
        epochs = 1, 
        verbose = 1, # progress bar
        # validation_split = 0.1,
        validation_data = (valid, np.array([[1,0] if label == '0' else [0,1] for label in valid_labels])),
        shuffle = True
    )
    pred = np.array(['0' if a > b else '1' for a, b in model.predict(valid)])
    model.save('ml-fmi-23-2020//Models//3//' + str(epoch) + '-' + str(sum(pred==valid_labels)))


# %%
# 4
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
model = Sequential()

model.add(Conv2D(16, (5,4), activation='relu', strides=1, padding='same', input_shape = train[0].shape))
model.add(Conv2D(16, (5,4), activation='relu', strides=1, padding='same'))
model.add(MaxPool2D((5,2)))
model.add(Conv2D(32, (5,4), activation='relu', strides=1, padding='same'))
model.add(MaxPool2D((5,2)))
model.add(Conv2D(32, (3,3), activation='relu', strides=1, padding='same'))
model.add(MaxPool2D((2,2)))
model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# %%
for epoch in range(15):
    model.fit(
        x = train, 
        y = np.array([[1,0] if label == '0' else [0,1] for label in train_labels]), 
        epochs = epoch+1, 
        initial_epoch = epoch,
        verbose = 1, # progress bar
        # validation_split = 0.1,
        validation_data = (valid, np.array([[1,0] if label == '0' else [0,1] for label in valid_labels])),
        shuffle = True
    )
    pred = np.array(['0' if a > b else '1' for a, b in model.predict(valid)])
    model.save('ml-fmi-23-2020//Models//4//' + str(epoch) + '-' + str(sum(pred==valid_labels)))


# %%
from keras.models import load_model
model = '7-stft-detailed//6'
epoch = '11-778' # epoch-validScore
saved_model = load_model('ml-fmi-23-2020//Models//' + model + '//' + epoch)
pred = np.array(['0' if a > b else '1' for a, b in saved_model.predict(valid)])
sum(pred==valid_labels)


# %%
pred = np.array(['0' if a > b else '1' for a, b in saved_model.predict(test)])
writeCsv(test_data_name, pred, 'ml-fmi-23-2020//Models//'+ model +'//predictions-' + model.replace('//', '_') + '_' + epoch + '.txt')


# %%
# # fit on both train and valid 
# train_valid = []
# train_valid.extend (train)
# train_valid.extend (valid)
# train_valid = np.array(train_valid)
# train_valid_labels = []
# train_valid_labels.extend (train_labels)
# train_valid_labels.extend (valid_labels)
# train_valid_labels = np.array(train_valid_labels)
# model.fit(
#     x = train_valid, 
#     y = np.array([[1,0] if label == '0' else [0,1] for label in train_valid_labels]), 
#     epochs = 1, 
#     verbose = 1, # progress bar
#     # validation_split = 0.1,
#     # validation_data = (valid, np.array([[1,0] if label == '0' else [0,1] for label in valid_labels])),
#     shuffle = True
# )


# %%
pred1 = np.array(['0' if a > b else '1' for a, b in model.predict(test)])


# %%
writeCsv(test_data_name, pred)


# %%
sum(pred!=pred1)


# %%
noModel = '7-stft_'

# for kaggle, comment them for other use
import os
try:
    os.mkdir('//kaggle//working//Models//')
except: pass
try:
    os.mkdir('//kaggle//working//Models//'+noModel)
except: pass
# end of for kaggle

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
model = Sequential()

model.add(Conv2D(16, (5,4), activation='relu', strides=1, padding='same', input_shape = train[0].shape))
model.add(Conv2D(16, (5,4), activation='relu', strides=1, padding='same'))
model.add(MaxPool2D((5,2)))
model.add(Conv2D(32, (5,2), activation='relu', strides=1, padding='same'))
model.add(MaxPool2D((5,2)))
model.add(Conv2D(32, (5,2), activation='relu', strides=1, padding='same'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
epoch = -1


# %%
for epoch in range(epoch+1, epoch+6+15):
    model.fit(
        x = train, 
        y = np.array([[1,0] if label == '0' else [0,1] for label in train_labels]), 
        epochs = epoch+1, 
        initial_epoch = epoch,
        verbose = 1, # progress bar
        # validation_split = 0.1,
        validation_data = (valid, np.array([[1,0] if label == '0' else [0,1] for label in valid_labels])),
        shuffle = True
    )
    pred = np.array(['0' if a > b else '1' for a, b in model.predict(valid)])
    # change where to save when not on kaggle
    model.save('//kaggle//working//Models//' + noModel + '//' + ('0' if epoch < 10 else '') + str(epoch) + '-' + str(sum(pred==valid_labels)))


# %%
from keras.models import load_model
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
valid_labels_val = np.array([[1,0] if label == '0' else [0,1] for label in valid_labels])


# %%
model = '7-stft-detailed//6_14-779'
epoch1 = model[-6:] # epoch-validScore
saved_model1 = load_model('ml-fmi-23-2020//Models//' + model + '//' + epoch1)
pred1_val = saved_model1.predict(valid)
pred1 = np.array(['0' if a > b else '1' for a, b in pred1_val])
print ("sum =", sum(pred1==valid_labels), "\nmae =", mae(valid_labels_val, pred1_val), "\nmse =", mse(valid_labels_val, pred1_val))


# %%
model = '7-stft-detailed//5_13-773'
epoch = model[-6:] #'13-767' # epoch-validScore
saved_model = load_model('ml-fmi-23-2020//Models//' + model + '//' + epoch)
pred_val = saved_model.predict(valid)
pred = np.array(['0' if a > b else '1' for a, b in pred_val])
print ("sum =", sum(pred==valid_labels), "\nmae =", mae(valid_labels_val, pred_val), "\nmse =", mse(valid_labels_val, pred_val))


# %%
model = '7-stft-detailed//0_13-767'
epoch = '13-767' # epoch-validScore
saved_model = load_model('ml-fmi-23-2020//Models//' + model + '//' + epoch)
pred_val = saved_model.predict(valid)
pred = np.array(['0' if a > b else '1' for a, b in pred_val])
print ("sum =", sum(pred==valid_labels), "\nmae =", mae(valid_labels_val, pred_val), "\nmse =", mse(valid_labels_val, pred_val))


# %%
model = '7-stft-detailed//6_14-779'
epoch = '11-778' # epoch-validScore
saved_model = load_model('ml-fmi-23-2020//Models//' + model + '//' + epoch)
pred_val = saved_model.predict(valid)
pred = np.array(['0' if a > b else '1' for a, b in pred_val])
print ("sum =", sum(pred==valid_labels), "\nmae =", mae(valid_labels_val, pred_val), "\nmse =", mse(valid_labels_val, pred_val))


# %%
model = '7-stft-detailed//9_11-773'
epoch = model[-6:] #'13-767' # epoch-validScore
saved_model = load_model('ml-fmi-23-2020//Models//' + model + '//' + epoch)
pred_val = saved_model.predict(valid)
pred = np.array(['0' if a > b else '1' for a, b in pred_val])
print ("sum =", sum(pred==valid_labels), "\nmae =", mae(valid_labels_val, pred_val), "\nmse =", mse(valid_labels_val, pred_val))


# %%
model = '11'
epoch4 = '20-766' # epoch-validScore
saved_model4 = load_model('ml-fmi-23-2020//Models//' + model + '//' + epoch4)
pred4_val = saved_model4.predict(valid)
pred4 = np.array(['0' if a > b else '1' for a, b in pred4_val])
print ("sum =", sum(pred4==valid_labels), "\nmae =", mae(valid_labels_val, pred4_val), "\nmse =", mse(valid_labels_val, pred4_val))


# %%
model = '3-stft_2'
epoch2 = '13-764' # epoch-validScore
saved_model2 = load_model('ml-fmi-23-2020//Models//' + model + '//' + epoch2)
pred2_val = saved_model2.predict(valid)
pred2 = np.array(['0' if a > b else '1' for a, b in pred2_val])
print ("sum =", sum(pred2==valid_labels), "\nmae =", mae(valid_labels_val, pred2_val), "\nmse =", mse(valid_labels_val, pred2_val))


# %%
model = '7-stft-detailed//1_08-749'
epoch = model[-6:] #'13-767' # epoch-validScore
saved_model = load_model('ml-fmi-23-2020//Models//' + model + '//' + epoch)
pred_val = saved_model.predict(valid)
pred = np.array(['0' if a > b else '1' for a, b in pred_val])
print ("sum =", sum(pred==valid_labels), "\nmae =", mae(valid_labels_val, pred_val), "\nmse =", mse(valid_labels_val, pred_val))


# %%
model = '4'
epoch3 = '4-753' # epoch-validScore
saved_model3 = load_model('ml-fmi-23-2020//Models//' + model + '//' + epoch3)
pred3_val = saved_model3.predict(valid)
pred3 = np.array(['0' if a > b else '1' for a, b in pred3_val])
print ("sum =", sum(pred3==valid_labels), "\nmae =", mae(valid_labels_val, pred3_val), "\nmse =", mse(valid_labels_val, pred3_val))


# %%
model = '7-stft-detailed//2_20-765'
epoch = model[-6:] #'13-767' # epoch-validScore
saved_model = load_model('ml-fmi-23-2020//Models//' + model + '//' + epoch)
pred_val = saved_model.predict(valid)
pred = np.array(['0' if a > b else '1' for a, b in pred_val])
print ("sum =", sum(pred==valid_labels), "\nmae =", mae(valid_labels_val, pred_val), "\nmse =", mse(valid_labels_val, pred_val))


# %%



# %%
model = '7-stft-detailed//4_24-774'
epoch = model[-6:] #'13-767' # epoch-validScore
saved_model = load_model('ml-fmi-23-2020//Models//' + model + '//' + epoch)
pred_val = saved_model.predict(valid)
pred = np.array(['0' if a > b else '1' for a, b in pred_val])
print ("sum =", sum(pred==valid_labels), "\nmae =", mae(valid_labels_val, pred_val), "\nmse =", mse(valid_labels_val, pred_val))


# %%
pred = np.array(['0' if a > b else '1' for a, b in saved_model.predict(test)])
writeCsv(test_data_name, pred, 'ml-fmi-23-2020//Models//'+ model +'//predictions-' + model.replace('//', '_') +'txt') # + '_' + epoch + '.txt')


# %%
pred.shape

