# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import csv
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from python_speech_features import mfcc, delta, logfbank
from random import randint
import librosa


FileNames = {"train":"//kaggle//input//ml-fmi-23-2020//train.txt", "valid":"//kaggle//input//ml-fmi-23-2020//validation.txt", "test":"//kaggle//input//ml-fmi-23-2020//test.txt", "pred":"//kaggle//output//ml-fmi-23-2020//predictions.txt"}


AudioFolders = {"train":"//kaggle//input//ml-fmi-23-2020//train//train//", "valid":"//kaggle//input//ml-fmi-23-2020//validation//validation//", "test":"//kaggle//input//ml-fmi-23-2020//test//test//"}

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
#         data.append(np.abs(fft(wav.read(AudioFolders[folderName]+dataName)[1])))
#         data.append(mfcc(wav.read(AudioFolders[folderName]+dataName)[1], numcep=12))#.reshape(-1))
        
    return np.array(data)


# %%
train_data_name, train_labels = readCsv(FileNames['train'], hasLables=True)
valid_data_name, valid_labels = readCsv(FileNames['valid'], hasLables=True)
test_data_name                = readCsv(FileNames['test'],  hasLables=False)
# writeCsv(train_data, train_labels)


# %%
train_data = getData('train', train_data_name)
valid_data = getData('valid', valid_data_name)
test_data  = getData('test',  test_data_name)
mi = min([train_data.min(), valid_data.min(), test_data.min()])
ma = max([train_data.max(), valid_data.max(), test_data.max()])
print (mi, ma)
train = (train_data - mi) / (ma-mi)
valid = (valid_data - mi) / (ma-mi)
test =  (test_data  - mi) / (ma-mi)
mi = min([train.min(), valid.min(), test.min()])
ma = max([train.max(), valid.max(), test.max()])
print (mi, ma)
train.shape, valid.shape, test.shape


# %%
initialShape = train.shape
train = train.reshape(initialShape[0], initialShape[1], initialShape[2], 1)
initialShape = valid.shape
valid = valid.reshape(initialShape[0], initialShape[1], initialShape[2], 1)
initialShape = test.shape
test = test.reshape(initialShape[0], initialShape[1], initialShape[2], 1)
train.shape, valid.shape, test.shape


# %%
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
model = Sequential()

noModel = '7-stft'

# for kaggle
import os
try:
    os.mkdir('//kaggle//working//Models//')
except: pass
try:
    os.mkdir('//kaggle//working//Models//'+noModel)
except: pass

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
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy', 'mae', 'mse'])

model.summary()
epoch = -1


# %%
for epoch in range(epoch+1, epoch+6+25):
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
    model.save('//kaggle//working//Models//' + noModel + '//' + ('0' if epoch < 9 else '') + str(epoch+1) + '-' + str(sum(pred==valid_labels)))

# %%
from keras.models import load_model
# select the epoch you want to load
# model =
epoch = '11-778' # epoch-validScore
saved_model = load_model('ml-fmi-23-2020//Models//' + model + '//' + epoch)
pred = np.array(['0' if a > b else '1' for a, b in saved_model.predict(valid)])
sum(pred==valid_labels)

# %%
pred = np.array(['0' if a > b else '1' for a, b in model.predict(test)])
writeCsv(test_data_name, pred)

