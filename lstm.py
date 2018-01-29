# -*- coding: utf-8 -*-
"""
	Sentiment Analysis with LSTMs
"""

import pandas
import collections
import keras
from keras.models import Sequential
from keras.models import h5py
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import SimpleRNN
from keras.optimizers import RMSprop
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import preprocessing  
from keras.utils import np_utils
import tensorflow as tf
from keras.layers import LSTM
from keras.layers import Masking
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding

wordsList = np.load('wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print ('Loaded the word vectors!')
print(len(wordsList))
print(wordVectors.shape)

#for testing
baseballIndex = wordsList.index('baseball')
print(wordVectors[baseballIndex])

#for testing
maxSeqLength = 10 #Maximum length of sentence
numDimensions = 300 #Dimensions for each word vector
firstSentence = np.zeros((maxSeqLength), dtype='int32')
firstSentence[0] = wordsList.index("i")
firstSentence[1] = wordsList.index("thought")
firstSentence[2] = wordsList.index("the")
firstSentence[3] = wordsList.index("movie")
firstSentence[4] = wordsList.index("was")
firstSentence[5] = wordsList.index("incredible")
firstSentence[6] = wordsList.index("and")
firstSentence[7] = wordsList.index("inspiring")
#firstSentence[8] and firstSentence[9] are going to be 0
print(firstSentence.shape)
print(firstSentence) #Shows the row index for each word
#with tf.Session() as sess:
#    print(tf.nn.embedding_lookup(wordVectors,firstSentence).eval().shape)
    

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"'", " ", string)
    string = re.sub(r"\[", " ", string)
    string = re.sub(r"\]", " ", string)
    return string.strip().lower()

# load dataset
def get_data(file_name, drop_list):
    data = pandas.read_csv(file_name)
    numWords = []
    word_freqs = collections.Counter()  #words frequency
    y_data1 = data[["Label"]]
    y_data = y_data1.values
    y_data = keras.utils.to_categorical(y_data)
    x_data_df = data.drop(drop_list,axis=1)
    x_data = x_data_df.values
    numFiles = len(x_data[:,0])
    numTops = len(x_data[0,:])
                         
    maxSeqLength = 600
    ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
    fileCounter = 0
    
    for i in range(numFiles):#numFiles
        line=''
        for j in range(numTops):#numTops
            try:
                line=line+' '+x_data[i,j][1:]
            except TypeError:
                print(x_data[i,j],i,j)
                continue
        #print(line)
        numWords.append(len(line.split()))
        indexCounter = 0
        cleanedLine = cleanSentences(line)#cleanSentences
        split = cleanedLine.split()
        for word in split:
            #print(word)
            word_freqs[word] += 1
            try:
                ids[fileCounter][indexCounter] = wordsList.index(word)
            except ValueError:
                #print(word)
                ids[fileCounter][indexCounter] = 400000-1 #Vector for unkown words
            indexCounter = indexCounter + 1
            if indexCounter >= maxSeqLength:
                break
        fileCounter=fileCounter+1
    
    import matplotlib.pyplot as plt
    plt.hist(numWords, 50)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.axis([100, 750, 0, 150])
    plt.show()
    print('The total number of files is', numFiles)
    print('The total number of words in the files is', sum(numWords))
    print('The average number of words in the files is', sum(numWords)/len(numWords))
    print('The different words is', len(word_freqs))
    
    np.save('X_Matrix', ids)
    np.save('Y_Matrix', y_data)
    return len(word_freqs), numWords, data.values, ids, y_data


drop_list = ['Date','Label'] #drop attributes

# get data
len_word_freqs, numWords, data, x_data, y_data = get_data("Combined_News_DJIA1.csv", drop_list) # you can comment this line for speed

print('Getting data done!!')
#x_data = np.load('X_Matrix.npy')
#y_data = np.load('Y_Matrix.npy')

#build rnn model lstm

# IF YOU WANT TO USE THE SAME INITIAL WEIGHTS
np.random.seed(1337) 
tf.set_random_seed(1337)

# 0.2 test data 0.8 train data
Xtrain, Xtest, ytrain, ytest = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

maxSeqLength = 600

Xtrain = Xtrain.reshape(-1, maxSeqLength, 1)
Xtest = Xtest.reshape(-1, maxSeqLength, 1)

len_word_freqs = 43747

nb_classes = 2
learning_rate = 1e-3
batch_size = 32
nb_epochs = 1000
hidden_units = 20

#early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, verbose=0, patience=120, mode='auto')
#checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=str("./best_model"), monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#
#model = Sequential()
#model.add(Masking(mask_value=0, input_shape=Xtrain.shape[1:]))
##        model.add(SimpleRNN(output_dim=hidden_units,
##                    activation='relu',
##                    input_shape=Xtrain.shape[1:], dropout=0))
#model.add(LSTM(output_dim=hidden_units,  activation='relu', recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=False, input_shape=Xtrain.shape[1:], dropout=0))
#       
##model.add(keras.layers.BatchNormalization())
##model.add(Dense(20, activation='relu'))
##model.add(keras.layers.Dropout(0.3))
#        
#model.add(Dense(nb_classes))
#model.add(Activation('softmax'))
#rmsprop = RMSprop(lr=learning_rate)
#model.compile(loss='categorical_crossentropy',optimizer=rmsprop,metrics=['accuracy'])
#model.fit(Xtrain, ytrain, batch_size=batch_size, nb_epoch=nb_epochs, 
#          verbose=1, validation_data=(Xtest, ytest),callbacks=[checkpoint_callback],shuffle=True)#,early_stop_callback

#model.load_weights(str("./best_model")) 

def recall(y_true, y_pred):
    index = np.where(y_true[:,1] == 1)
    #print "index",index
    positive_sample = y_true[index]
    predicts_sample = y_pred[index]
    true_positive = predicts_sample[np.where(predicts_sample[:,1] >= 0.5)]
    #print (len(true_positive))
    #print (len(positive_sample))
    return len(true_positive)/len(positive_sample)

#score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
#y_pred = model.predict(Xtest)
#best_recall = recall(ytest, y_pred)
#
#print("\nTest score: %.3f, accuracy: %.3f, recall: %.3f" % (score, acc, best_recall))
#print('{}   {}      {}'.format('Predict','Truth','Sentence'))
#for i in range(3):
#    idx = np.random.randint(len(Xtest))
#    xtest = Xtest[idx].reshape(1,40)
#    ylabel = ytest[idx]
#    ypred = model.predict(xtest)[0][0]
#    sent = " ".join([wordsList[x] for x in xtest[0] if x != 0])
#    print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))
