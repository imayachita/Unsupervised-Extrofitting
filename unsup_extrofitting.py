#This code is written based on Extrofitting implementation on https://github.com/HwiyeolJo/Extrofitting
#Reference paper: Extrofitting: Enriching Word Representation and its Vector Space with Semantic Lexicons
#arXiv:1804.07946


from __future__ import print_function
import math
import numpy as np
import re
from copy import deepcopy
from tqdm import tqdm_notebook
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from gensim.models import FastText, KeyedVectors
from sklearn.decomposition import PCA
from sklearn.utils.multiclass import unique_labels
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from scipy import linalg
from numpy.linalg import inv
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from gensim.models import Word2Vec

### Hyperparameter
WordDim = 300
ExpandDim = 100
NormRead = True
ExpandNorm = True
#####

#unsupervised extrofitting

def read_word_vecs(filename):
    print("Vectors read from", filename)
    wordVectors = {}
    fileObject = open(filename, 'r')
    fileObject.readline() # For handling First Line
    for line in fileObject:
        line = line.strip().lower()
        word = line.split()[0]
        wordVectors[word] = np.zeros(len(line.split())-1, dtype=np.float64)
        vector = line.split()[1:]
        if len(vector) == WordDim:
            for index, vecVal in enumerate(vector):
                wordVectors[word][index] = float(vecVal)
            if NormRead:
                wordVectors[word] = wordVectors[word] / math.sqrt((wordVectors[word]**2).sum() + 1e-5)
    return wordVectors



'''
get the vectors for each word, save it in a dictionary
'''
model_dir="/home/codes/Extrofitting-master/glove.txt"


wordVecs = read_word_vecs(model_dir)
newWordVecs = deepcopy(wordVecs)
n_epochs=5


for epoch in range(n_epochs):
    print('Starting Epoch: ', epoch)
    '''
    create a new numpy array to store the word vectors from the dictionary
    '''
    wordVec_np=[]
    for k in newWordVecs.keys():
        wordVec_np.append(newWordVecs[k])
    wordVec_np = np.array(wordVec_np) #wordVec_np is an numpy array containing word vectors
    num_vocab = len(wordVec_np)
    print('vocab length: ', num_vocab)

    '''
    decompose word vectors wordVec_np using SVD
    '''
    U, S, V = randomized_svd(wordVec_np,
                                  n_components=WordDim,
                                  n_iter=5,
                                  random_state=None)

    print('U, S, V: ', U.shape,S.shape,V.shape)


    '''
    get the semantic representation for each word with matrix multiplication U and S
    '''
    US = np.dot(U,np.diag(S))
    print('US Shape', US.shape) #US shape is [n_features,n_components]


    '''
    create new dictionary to store the US representation called decomposed
    '''
    decomposed={}
    for i,word in enumerate(newWordVecs.keys()):
        decomposed[word]=US[i]

    '''
    expand the word vector dimension
    '''
    ExpandDim=2
    for w in newWordVecs.keys():
        for i in range(ExpandDim):
            decomposed[w] = np.hstack((decomposed[w], np.mean(decomposed[w])))
        decomposed[w] = np.hstack((decomposed[w], np.zeros(1)))
    len_expanded=len(decomposed[w])
    print('expanded vector: ', len_expanded)


    '''
    save the decomposed matrix to a text file to be opened using gensim later
    '''
    def print_dict(filename,dic):
        with open(filename, 'w') as file:
            file.write(str(num_vocab)+' '+str(len_expanded)+'\n')
            for key in dic:
                file.write(key+' ')
                for element in dic[key]:
                    file.write(str(element)+' ')
                file.write('\n')

    decomposed_file='decomposed.txt'
    print_dict(decomposed_file,decomposed)

    '''
    calculating cosine similarity and classifying
    '''
    labels=[]
    x=[]
    wordidx=0

    print('calculating cosine similarity and classifying...')

    wordVec_np = []
    for k in decomposed.keys():
        wordVec_np.append(decomposed[k])
    wordVec_np = np.array(wordVec_np)

    model = KeyedVectors.load_word2vec_format(decomposed_file)
    topn=20
    T=0.7 #threshold of cosine similarity
    wordidx=0
    for word in decomposed.keys():
        wordidx+=1
        if decomposed[word][-1]==0:
            decomposed[word][-1]=wordidx
            sim = model.most_similar(word,topn=topn)
#             print('len sim: ', len(sim))
            max_words = np.minimum(len(sim),topn)
#             print(max_words)
            for i in range(max_words):
                sim_word=sim[i][0]
                sim_score=sim[i][1]
                if sim_score>=T:
    #                 print(word,sim_word,sim_score)
                    decomposed[sim_word][-1]=wordidx
    # print('decomposed matrix: ', decomposed)

    '''
    Re-assign wordVec_np with the new vectors after expanding and classifying the words
    '''
    wordVec_np = []
    for k in decomposed.keys():
        wordVec_np.append(decomposed[k])
    wordVec_np = np.array(wordVec_np)
    # print(wordVec_np[0])

    '''
    I don't know that this part is doing, just following the paper
    '''
    if ExpandNorm:
        wordVec_np[:,-ExpandDim:-1] \
        = wordVec_np[:,-ExpandDim:-1] / np.sqrt(np.sum(wordVec_np[:,-ExpandDim:-1]**2, axis=0) + 1e-5)

    # print('expand norm: ', wordVec_np[:,-ExpandDim:-1])
    # print(wordVec_np[0])

    print('labels: ', wordVec_np[:,-1])
    # duplicates = [item for item, count in Counter(wordVec_np[:,-1]).items() if count > 1]
    # print('DUPLICATES: ', duplicates)

    print('num_classes: ', len(unique_labels(wordVec_np[:,-1])))
    print('n_features: ', wordVec_np[:,:-1].shape)

    '''
    reduce dimensionality with PCA
    '''
    pca = PCA(n_components=WordDim)
    wordVec_np = pca.fit_transform(wordVec_np[:,:-1], wordVec_np[:,-1])

    print('Length after PCA: ', len(wordVec_np[0]))

#     '''
#     reduce dimensionality with LDA
#     '''
#     lda = LinearDiscriminantAnalysis(n_components=10)
#     wordVec_np = lda.fit_transform(wordVec_np[:,:-1], wordVec_np[:,-1])

    '''
    Re-assign newWordVecs with new wordVec_np after dimensionality reduction
    '''
    for i, k in enumerate(newWordVecs.keys()):
        newWordVecs[k] = wordVec_np[i]


print_word_vecs(newWordVecs, 'Unsup_extro_glove' + str(ExpandDim) + '_' + str(n_epochs) + '.txt')
