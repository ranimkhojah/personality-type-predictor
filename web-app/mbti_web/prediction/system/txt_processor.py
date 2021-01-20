# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'web-app\\mbti_web\prediction\system'))
	print(os.getcwd())
except:
	pass
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle


#### AUTHOR OF THIS FILE: DUY NGUYEN NGOC AND RANIM KHOJAH
#### RANIM KHOJAH WORKED ON THE DEFINITION OF THE METHODS split_words(), lemmatize() and tokenize()
#### DUY NGUYEN NGOC WORKD ON THE REST AND THE IMPROVEMENT OF THE ABOVE METHODS.

# %%
# Split text into tuple of words
def split_words(posts):
    tokenized_posts = []
    for row in posts:
        token = word_tokenize(row)
        if token != '':
            tokenized_posts.append(token)
#    print('tokenized posts', tokenized_posts)
    return tokenized_posts


# %%
def get_word_num(posts):
    return len(tokenize(posts))


# %%
#Get POS-tag based on treebank definition.
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# %%
#Lemmatize the tuple of words and return the lemmatized sentence/post (Read the final report for more information)
def lemmatize(tokenized_posts):
    import nltk
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    lemmatizer = WordNetLemmatizer()
    lemmatized_posts = []

    print("For Loop incoming")
    for sentence in tokenized_posts:
        tagged = pos_tag(sentence)
        lemmatized_sentence = []
        for word, tag in tagged:
            
            wntag = get_wordnet_pos(tag)
            if wntag is None:
                
                lemmatized_sentence.append(word)
            else:
    
                lemmatized_sentence.append(lemmatizer.lemmatize(word, pos=wntag))

        
        lemmatized_sentence = " ".join(lemmatized_sentence)
        lemmatized_posts.append(lemmatized_sentence)
    print("out of the loop")
    return lemmatized_posts


# %%

#Create a tokenizer object of the whole text corpus and save the tokenizer so we can use it for prediction and evaluation
def create_tokenizer(data, path):
    all_words = [word for tokens in data["tokens"] for word in tokens]
    VOCAB = sorted(list(set(all_words)))
    tokenizer = Tokenizer(num_words=len(VOCAB), lower=True, char_level=False)
    tokenizer.fit_on_texts(data["lemmatized_posts"])
    
    with open(path + 'tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return tokenizer


# %%

#Fit the data on the tokenizer and vectorize the text into sequence of integers
def tokenize(data, tokenizer, sequence_length):
    sequences = tokenizer.texts_to_sequences(data["lemmatized_posts"])
    #print(sequences)
    cnn_data = pad_sequences(sequences, maxlen=sequence_length)
    return cnn_data


# %%
def get_vocab_size(tokenizer):
    vocab_size = len(tokenizer.word_index) + 1  
    return vocab_size



# %%
#Create a new target column that only contains the specified character belonging to the full 4 character personality type
def split_target_variable(data, character):
    data['character'] = [characters[character] for characters in data.type]
    return data


# %%
#Create a new target column of the binarized single character target
def binarize_target_variable(data):
    mapping = {'I': 1, 'N': 1, 'F': 1, 'P': 1, 'E': 0, 'S': 0, 'T': 0, 'J': 0}
    data['binarized_target'] = data['character'].apply(lambda s: mapping.get(s) if s in mapping else s)
    return data


# %%
#Undersample the majority ground truth value to have the same size of the minority
def undersample(data):
    major = 0
    if data.groupby('binarized_target').count().sort_values('posts').index[-1] == 1:
        major = 1
    minor = 1 - major
    no_minor = len(data[data['binarized_target'] == minor])
    
    major_indices = data[data.binarized_target == major].index
    random_indices = np.random.choice(major_indices,no_minor, replace=False)
    minor_indices = data[data.binarized_target == minor].index
    
    under_sample_indices = np.concatenate([minor_indices,random_indices])
    under_sample = data.loc[under_sample_indices]
    return under_sample


### ALL FUNCTIONS BELOW CAN BE USED FOR FUTURE USE. THE FUNCTIONS CONTAIN FEATURIZATION OF THE TABULAR DATA (no. of CAPS, TF-IDF, etc.)
### WE CAN FIT 2 TYPES OF INPUT: TABULAR INPUT AND TEXT SEQUENCE FOR BETTER ACCURACY.



# %%
def tabular_features(tabular_data, tabular_train, tabular_test):
    tabular_train_data = pd.DataFrame()
    tabular_test_data = pd.DataFrame()
    for i in np.arange(1,4):
        tfidf = TfidfVectorizer(stop_words='english',ngram_range=(i,i), decode_error='replace', max_features = 100000)
        tabular_word_data = tfidf.fit(tabular_data['posts'].values.astype('U'))
        tabular_word_train = tfidf.fit_transform(tabular_train['posts'].values.astype('U'))
        tabular_word_test = tfidf.transform(tabular_test['posts'].values.astype('U'))


        tsvd = TruncatedSVD(n_components=500, algorithm='arpack', random_state=500)
        tabular_wordie_train = tsvd.fit_transform(tabular_word_train)
        tabular_wordie_test = tsvd.transform(tabular_word_test)
        tabular_wordie_train_df = pd.DataFrame(tabular_wordie_train,
                                        columns=[str(i)+'_'+str(b) for b in np.arange(1,tabular_wordie_train.shape[1]+1)])
        tabular_wordie_test_df = pd.DataFrame(tabular_wordie_test,
                                       columns=[str(i)+'_'+str(b) for b in np.arange(1,tabular_wordie_test.shape[1]+1)])
        tabular_train_data = pd.concat([tabular_train_data,tabular_wordie_train_df], axis=1)
        tabular_test_data = pd.concat([tabular_test_data,tabular_wordie_test_df], axis=1)
    return tabular_train_data, tabular_test_data


# %%
def tabular_scaler(tabular_train_data, tabular_test_data):
    scaler = MinMaxScaler()
    tabular_train_data = pd.DataFrame(scaler.fit_transform(tabular_train_data),
                                      columns=tabular_train_data.columns, index=tabular_train_data.index)
    tabular_test_data = pd.DataFrame(scaler.transform(tabular_test_data),
                                      columns=tabular_test_data.columns, index=tabular_test_data.index)
    return tabular_train_data, tabular_test_data


# %%
def chi2_features(tabular_train_data, tabular_test_data, tabular_y_train):
    chi2_features = SelectKBest(chi2, k = 100) 
    tabular_train_best_data = pd.DataFrame(chi2_features.fit_transform(tabular_train_data, tabular_y_train))
    tabular_test_best_data = pd.DataFrame(chi2_features.transform(tabular_test_data))
    return tabular_train_best_data, tabular_test_best_data


# %%
#check the % of the vocabulary covered by the pretrained model
#nonzero_elements = np.count_nonzero(np.count_nonzero(get_embedding_matrix(50, get_vocab_size(posts),"glove.6B.50d.txt" ), axis=1))
#nonzero_elements / get_vocab_size()


# %%


