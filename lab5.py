# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
# from librosa import fft_frequencies
# np.fft.fft

# %%
# 5.2
def standard_norm (data):
    m = np.mean(data)
    d = np.std (data)
    return np.divide (np.add(data, -m), d) # (data-m)/d

def min_max_norm (data):
    mi = np.min(data)
    ma = np.max(data)
    if ma == mi: # all valus will become 0 anyway
        ma = mi+1 
    return np.divide (np.add(data, -mi), ma-mi)

def l1_norm (data):
    return np.divide (data, np.sum(np.abs(data))) # data / sum(abs(data))

def l2_norm (data):
    return np.divide (data, np.sqrt(np.sum(np.power(data, 2)))) # data / (sum(data**2)**0.5)

def normalize_data(train_data, test_data, normtype=None):
    if normtype == 'standard':
        return np.array([standard_norm(data) for data in train_data]), np.array([standard_norm(data) for data in test_data])
    if normtype == 'min_max':
        return np.array([min_max_norm(data) for data in train_data]), np.array([min_max_norm(data) for data in test_data])
    if normtype == 'l1':
        return np.array([l1_norm(data) for data in train_data]), np.array([l1_norm(data) for data in test_data])
    if normtype == 'l2':
        return np.array([l2_norm(data) for data in train_data]), np.array([l2_norm(data) for data in test_data])
    # else: None or Unknown type
    return train_data, test_data


# %%
x_train = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]], dtype=np.float64)
x_test = np.array([[-1, 1, 0]], dtype=np.float64)
normalize_data (x_train, x_test, 'min_max')


# %%
[(np.mean(data),np.std(data)) for data in normalize_data (x_train, x_test, 'standard')[0]]


# %%
train_sentances = np.load('data-lab5/training_sentences.npy', allow_pickle=True)
train_labels    = np.load('data-lab5/training_labels.npy', allow_pickle=True).astype(np.bool)
test_sentances  = np.load('data-lab5/test_sentences.npy', allow_pickle=True)
test_labels     = np.load('data-lab5/test_labels.npy', allow_pickle=True).astype(np.bool)
print (train_sentances, '\n', train_labels)


# %%
# 5.3
class BagOfWords:
    
    def __init__ (self):
        self.vocab = {}
        self.cuvs  = []
    
    def build_vocabulary (self, data):
        for prop in data:
            for cuv in prop:
                if not cuv in self.vocab:
                    self.vocab[cuv] = len(self.cuvs)
                    self.cuvs.append(cuv) # ids starts with 0
    # 5.4
    def get_features(self, data):
        return np.array([np.bincount([self.vocab[cuv] for cuv in prop], minlength=len(self.cuvs)) for prop in data] ).astype(np.int).reshape((len(data), len(self.cuvs)))


# %%
# 5.5
train = BagOfWords()
train.build_vocabulary (train_sentances)
x_train = train.get_features(train_sentances)

test = BagOfWords()
test.build_vocabulary (test_sentances)
x_test = test.get_features(test_sentances)

x_train_norm, x_test_norm = normalize_data(x_train, x_test, 'l2')
x_train_norm, x_test_norm


# %%
from sklearn.svm import SVC
svm_train_data   = np.load('data-lab5/svm_train_data.npy', allow_pickle=True)
svm_train_labels = np.load('data-lab5/svm_train_labels.npy', allow_pickle=True).astype(np.bool)


# %%
svc = SVC (1, 'linear')
svc.fit(svm_train_data, svm_train_labels)
svc.score(svm_train_data, svm_train_labels)


# %%



# %%


