#Unsupervised Extrofitting
#This code is the implementation of paper https://arxiv.org/abs/1804.07946
#Written based on https://github.com/HwiyeolJo/Extrofitting


from sklearn.decomposition import TruncatedSVD
from scipy import linalg
from numpy.linalg import inv
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from gensim.models import Word2Vec
import numpy as np
import math
from copy import deepcopy
from gensim.models import FastText, KeyedVectors
from sklearn.utils.multiclass import unique_labels
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from tqdm import tqdm
import argparse


def print_dict(filename,dic):
    '''
    save the decomposed matrix to a text file to be opened using gensim later
    Args:
        filename: output file name
        dic: dictionary that has the word as the key and its vector as the element
    '''

    len_expanded = len(dic[list(dic.keys())[0]])
    with open(filename, 'w') as file:
        file.write(str(num_vocab)+' '+str(len_expanded)+'\n')
        for key in dic:
            file.write(key+' ')
            for element in dic[key]:
                file.write(str(element)+' ')
            file.write('\n')


def SVD(wordVec_np,newWordVecs,ExpandDim=2):
    '''
    decompose word vectors wordVec_np using SVD
    Args:
        wordVec_np: word vectors in numpy array
        newWordVecs: dictionary that has the word as the key and its vector as the element
        ExpandDim: dimension expansion
    Returns:
        decomposed: numpy array that contains the decomposed word vectors
    '''
    U, S, V = randomized_svd(wordVec_np,
                                  n_components=WordDim,
                                  n_iter=5,
                                  random_state=None)

    # print('U, S, V: ', U.shape,S.shape,V.shape)

    #get the semantic representation for each word with matrix multiplication U and S
    US = np.dot(U,np.diag(S))
    # print('US Shape', US.shape) #US shape is [n_features,n_components]

    #create new dictionary to store the US representation called decomposed
    decomposed={}
    for i,word in enumerate(newWordVecs.keys()):
        decomposed[word]=US[i]

    #expand the word vector dimension
    for w in newWordVecs.keys():
        for i in range(ExpandDim):
            decomposed[w] = np.hstack((decomposed[w], np.mean(decomposed[w])))
        decomposed[w] = np.hstack((decomposed[w], np.zeros(1)))
    len_expanded=len(decomposed[w])
    print('expanded vector: ', len_expanded)
    return decomposed


def most_similar(decomposed,decomposed_file,T):
    '''
    Calculating cosine similarity with gensim most_similar method
    Put the words that have coefficient > T in the same class
    We load word embedding vector from text file because we want to use gensim most_similar method

    Args:
        decomposed: decomposed word vectors
        decomposed_file: txt file that contains the US decomposed word vector
        T: gensim most_similar method coefficient threshold
    Returns:
        decomposed: the new decomposed word vectors that classified words to new classes
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
    T=0.7
    wordidx=0
    for word in decomposed.keys():
        wordidx+=1
        if decomposed[word][-1]==0:
            decomposed[word][-1]=wordidx
            sim = model.most_similar(word,topn=topn)
            for i in range(topn):
                sim_word=sim[i][0]
                sim_score=sim[i][1]
                if sim_score>=T:
                    decomposed[sim_word][-1]=wordidx
    return decomposed


def reduce_dim_PCA(n_comp,wordVec_np):
    '''
    reduce dimensionality with PCA
    Args:
        n_comp: output dimension of PCA
        wordVec_np: word vectors in numpy array in expanded dimension
    Returns:
        wordVec_np: word vectors in numpy array after PCA
    '''
    pca = PCA(n_components=n_comp)
    wordVec_np = pca.fit_transform(wordVec_np[:,:-1], wordVec_np[:,-1])

    print('Length after PCA: ', len(wordVec_np[0]))
    return wordVec_np


def reduce_dim_LDA(n_comp,wordVec_np):
    '''
    reduce dimensionality with LDA
    Args:
        n_comp: output dimension of PCA
        wordVec_np: word vectors in numpy array in expanded dimension
    Returns:
        wordVec_np: word vectors in numpy array after LDA
    '''
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    wordVec_np = lda.fit_transform(wordVec_np[:,:-1], wordVec_np[:,-1])
    return wordVec_np


def read_word_vecs(filename):
    '''
    Read word vectors
    Args:
        filename: input
    Returns:
        wordVectors: dictionary that has the word as the key and its vector as the element
    '''
    print("Vectors read from: ", filename)
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


def print_word_vecs(wordVectors, outFileName):
    '''
    Print word vectors to a text file
    Args:
        wordVectors: dictionary of word vectors to print
        outFileName: output file name
    '''
    print('Writing down the vectors in', outFileName)
    outFile = open(outFileName, 'w')
    outFile.write(str(len(wordVectors)) + ' ' + str(WordDim) + '\n')
    pbar = tqdm(total = len(wordVectors), desc = 'Writing')
    for word, values in wordVectors.items():
        pbar.update(1)
        outFile.write(word+' ')
        for val in wordVectors[word]:
            outFile.write('%.5f' %(val)+' ')
        outFile.write('\n')
    outFile.close()
    pbar.close()


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", type=str, required=True, help="model directory")
    ap.add_argument("-epoch", type=int, required=True, help="number of epochs")
    ap.add_argument("-o", type=str, required=True, help="output file name")


    args = ap.parse_args()
    model_dir = args.m
    n_epochs = args.epoch
    outfile = args.o

    # model_dir="/home/inneke/Documents/D_drive/Facilities/facilities_model/model_fasttext_retrofit_wordnet.txt"
    ExpandNorm = True
    NormRead = True
    T = 0.4 #threshold of cosine similarity
    WordDim = 300
    ExpandDim = 100


    #get the vectors for each word, save it in a dictionary
    wordVecs = read_word_vecs(model_dir)
    newWordVecs = deepcopy(wordVecs)


    for epoch in range(n_epochs):
        print('Starting Epoch', epoch+1, '..')
        #create a new numpy array to store the word vectors from the dictionary
        wordVec_np=[]
        for k in newWordVecs.keys():
            wordVec_np.append(newWordVecs[k])
        wordVec_np = np.array(wordVec_np)
        #wordVec_np is an numpy array containing word vectors
        num_vocab = len(wordVec_np)
        # print('vocab length: ', num_vocab)

        #decompose the word vector using SVD and expand the dimension
        decomposed = SVD(wordVec_np,newWordVecs,ExpandDim=ExpandDim)
        decomposed_file='decomposed.txt'
        print_dict(decomposed_file,decomposed)
        decomposed = most_similar(decomposed,decomposed_file,T)


        #Re-assign wordVec_np with the new vectors after expanding and classifying the words
        wordVec_np = []
        for k in decomposed.keys():
            wordVec_np.append(decomposed[k])
        wordVec_np = np.array(wordVec_np)


       if ExpandNorm:
            wordVec_np[:,-ExpandDim:-1] \
            = wordVec_np[:,-ExpandDim:-1] / np.sqrt(np.sum(wordVec_np[:,-ExpandDim:-1]**2, axis=0) + 1e-5)

        # print('labels: ', wordVec_np[:,-1])
        # print('num_classes: ', len(unique_labels(wordVec_np[:,-1])))
        # print('n_features: ', wordVec_np[:,:-1].shape)


        #reduce dimensionality with PCA. The original paper uses LDA.
        #the downside of LDA is that we can't control the final dimension, it is controlled by the rank determined by LDA algorithm itself
        # wordVec_np = reduce_dim_PCA(WordDim,wordVec_np)
        wordVec_np = reduce_dim_LDA(WordDim,wordVec_np)



        #Re-assign newWordVecs with new wordVec_np after dimensionality reduction
        for i, k in enumerate(newWordVecs.keys()):
            newWordVecs[k] = wordVec_np[i]

    print_word_vecs(newWordVecs, outfile)
