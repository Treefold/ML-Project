# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import csv
from scipy.io import wavfile
from random import randint
import librosa
from librosa import display
import matplotlib.pyplot as plt

FileNames = {"train":"ml-fmi-23-2020//train.txt", "valid":"ml-fmi-23-2020//validation.txt", "test":"ml-fmi-23-2020//test.txt", "ex":"ml-fmi-23-2020//sample_submission.txt","pred":"ml-fmi-23-2020//predictions.txt"}

AudioFolders = {"train":"ml-fmi-23-2020//train//train//", "valid":"ml-fmi-23-2020//validation//validation//", "test":"ml-fmi-23-2020//test//test//"}

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


# %%
# the len of each audio file is 16000 
train_data, train_labels = readCsv(FileNames['train'], hasLables=True)
valid_data, valid_labels = readCsv(FileNames['valid'], hasLables=True)
test_data                = readCsv(FileNames['test'],  hasLables=False)
# writeCsv(train_data, train_labels)


# %%
def plotdata (dataLen = 10, t=None):
    if t not in ['0', '1', None]:
        raise Exception ("UnknownType")
    for i in range (dataLen):
        curr = randint(0, len(train_data)-1)
        if t is not None:
            while train_labels[curr] != t:
                curr = randint(0, len(train_data)-1)
        
        data = wavfile.read(AudioFolders['train']+train_data[curr])[1]

        absdata = np.abs(data)
        sorteddata = np.sort(absdata)
        floatdata = data.astype(np.float)

        fig, axis = plt.subplots(2,4)
        plt.subplots_adjust(wspace=1)
        axis[0,1].set_title (str(i) + '. Audio signal Data in time ' + str(curr) + ' - ' + ("Mask" if train_labels[curr]=='1' else "NoMask"),size=16)

        axis[0,0].plot(data)
        axis[1,0].hist(data)
        axis[0,1].plot(absdata)
        axis[1,1].hist(absdata)
        axis[0,2].plot(sorteddata)
        axis[1,2].hist(sorteddata)

        plt.show()

def dataHist (audioName, folder, show=True, isAbs=False):
    data = []
    for name in audioName:
        data.extend(wavfile.read(AudioFolders[folder]+name)[1])
    if isAbs:
        data = np.abs(data)
    plt.hist(data)
    plt.title('All Hist for ' + folder,size=16)
    if show:
        plt.show()
    return data

def dataHistAll (audioNames, folderKeys, isAbs=False):
    data = []
    for i in range (len(audioNames)):
        data.extend(dataHist(audioNames[i], folderKeys[i], isAbs=isAbs))
    plt.hist(data)
    plt.title('Hist over all',size=16)
    plt.show()

def spectrogram (dataLen = 10, t=None):
    if t not in ['0', '1', None]:
        raise Exception ("UnknownType")
    for i in range (dataLen):
        curr = randint(0, len(train_data)-1)
        if t is not None:
            while train_labels[curr] != t:
                curr = randint(0, len(train_data)-1)
        
        # data = wavfile.read(AudioFolders['train']+train_data[randint(0, len(train_data)-1)])[1].astype(np.float)
        x , sr = librosa.load(AudioFolders['train']+train_data[curr])
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(14, 5))
        plt.title(str(i) + '. Audio signal Data in time ' + str(curr) + ' - ' + ("Mask" if train_labels[curr]=='1' else "NoMask"),size=16)
        display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar()
        plt.show()
        plt.figure(figsize=(14, 5))
        plt.title(str(i) + '. Audio signal Data in time ' + str(curr) + ' - ' + ("Mask" if train_labels[curr]=='1' else "NoMask"),size=16)
        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar()
        plt.show()

# dataHistAll ([train_data, valid_data, test_data], list(AudioFolders.keys()))
# plotdata()


# %%
spectrogram()


# %%
data = wavfile.read(AudioFolders['train']+train_data[randint(0, len(train_data)-1)])[1].astype(np.float)
print(librosa.core.stft(data))
plt.plot(librosa.core.stft(data))
plt.show()
print(librosa.feature.melspectrogram(data))
plt.plot(librosa.feature.melspectrogram(data))
plt.show()


# %%
print(librosa.core.stft(data))
plt.plot(librosa.core.stft(data))
plt.show()
print(librosa.feature.melspectrogram(data))
plt.plot(librosa.feature.melspectrogram(data))
plt.show()

