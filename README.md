# Unsupervised-Extrofitting
This code is the implementation of this paper 
Extrofitting: Enriching Word Representation and its Vector Space with Semantic Lexicons by Hwiyeol Jo, Stanley Jungkyu Choi (https://arxiv.org/abs/1804.07946)

Code is written based on https://github.com/HwiyeolJo/Extrofitting with some modifications:
1. Function to reduce dimensionality with PCA instead of LDA
2. Use Gensim most_similar method to find the top-N most associated words

# Setup
The code is written in Python 3.6
To install the required packages:
```
pip install -r requirements.txt
```


How to run the code:
```
python unsupervised_extrofitting.py -m [original_model_file] -o [output_file] -e [number_of_epochs] 
```

Example:
``` 
python unsupervised_extrofitting.py -m model_fasttext.txt -o model_fasttext_unsupextrofit.txt -e 5 
```
