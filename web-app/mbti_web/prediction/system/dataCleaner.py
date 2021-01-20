#!/usr/bin/env python
#this specifies the coding of this file for handling letters of different alphabets
# -*- coding: utf-8 -*-

import pandas as pd
import re
try:
    from . import db_communicator
except Exception:
    import db_communicator

# Cleans dataset from URLs - author: Salvatore
def cleanURL(data):
    regex=re.compile (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return regex.sub('', data)


# Cleans dataset from alphabets besides latin - author: Salvatore
def filterLatinAlphabet(data):
    regex = re.compile(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]')
    return regex.sub('', data)


### Made by Hakim El Amri
#This function will take a string and remove everything that isn't a alphabetic character
def remove_punc(post):
    new_post = ''
    for ch in post:
        new_post += checkForCH(ch)
    return new_post

### Made by Hakim El Amri
# This is a helper function for the remove_punc function
def checkForCH(ch):
    CharToSpace = ',-'
    if ch.isalpha():
        return ch
    elif ch in (CharToSpace) or ch.isspace():
        return ' '
    else:
        return ''

def load_vocabulary(): # loads words from English vocabulary - Author: Salvatore

    try:
        with open('prediction/system/words_alpha.txt') as word_file:
            valid_words = set(word_file.read().split())
    except Exception:
        with open('words_alpha.txt') as word_file:
            valid_words = set(word_file.read().split())
    
    return valid_words

def clean_nonenglish_words(sentence, vocabulary=load_vocabulary()): #Author: Salvatore
    for word in sentence:
        if word.casefold() not in vocabulary or word == "w":
            sentence.remove(word)

def clean_nonenglish_words_printing(sentence, vocabulary=load_vocabulary()):# Author: Salvatore
    for word in sentence:
        if word.casefold() not in vocabulary:
            sentence.remove(word)
            print(word, " not english")
        if word.casefold() in vocabulary:
            print(word, " ENGLISH")

### Made by Hakim El Amri
# This function will take a Dataframe object and itterate over it...
# and return a new Dataframe object that has the post seperated by "|||"
# The data variable in the function is the Dataframe
# The rows variable in the function call is how many rows you want to go over.
# Negative row input will make it go thourgh the whole dataset
def seperat_data(data, row = 0):

    ### Makes a new dataframe with colums that will be returned at the end.
    new_data = pd.DataFrame([], columns=("type", "posts"))
    current = 0
    loop = 0
    # checks if the row variable is given and also if it is not to big, else it just keeps going
    if row > 0 and row <= len(data):
        data = data.head(row)


    max = len(data)
    longest_word = 0
    smallest_word = 10000
    for obj in data.iterrows():
        ### The lines below show which datapoint your currently working on out of all of them.
        current += 1
        
        if current == 75:
            loop +=1
            print("Cleaning post nr", (current*loop) ,"out of",max)
            current = 0
        ### Here I grab the type of the row and then I split the posts into a list, seperating at |||
        mbti_type = obj[1].at["type"]

        posts = obj[1].at["posts"]

        ### In this for loop I make dictonaries using the type as the key and then I loop over the list...
        ### the all the posts and I check which of them is also long enough to be included
       
        posts = filterLatinAlphabet(posts)
        

        posts = cleanURL(posts)
        
        sentence = remove_punc(posts)
        
        

        sentence = sentence.split()
        

        clean_nonenglish_words(sentence)
        

        clean_nonenglish_words(sentence)
        

        clean_nonenglish_words(sentence)
        

        if len(sentence) >= 100:
            if len(sentence) > longest_word:
                longest_word = len(sentence)

            elif len(sentence) < smallest_word:
                smallest_word = len(sentence)
            
            sentence=join_string(sentence)
        
            dict = { "type" : mbti_type , "posts" : sentence}
            new_data = new_data.append(other= dict, ignore_index= True)
    
    return new_data

### Made by Hakim El Amri
# This function will load the base CSV file and then will go over it a specified amount of times.
# The rows variable in the function call is how many rows you want to go over.
def cleaning_data(data, rows = 0):
    #this is the function call to seperate the posts in the dataset
    data = seperat_data(data, rows)
    
    #This is how I save a new file that has been worked on.
    file_name = "MBTI_Clean_test.csv"
    data.to_csv(file_name, index = False)


    #This will print the head of the dataframe if uncommented
    # print(data.head)

def join_string(list_string):
    # Join the string based on space delimiter
    string = ' '.join(list_string)

    return string

def clean_sentence(sentence):
    sentence = filterLatinAlphabet(sentence)
    sentence = cleanURL(sentence)
    sentence = remove_punc(sentence)
    sentence=sentence.split()
    clean_nonenglish_words(sentence)
    clean_nonenglish_words(sentence)
    sentence = join_string(sentence)
    return sentence
