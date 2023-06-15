import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
import json
import pickle
import numpy as np
stemmer=PorterStemmer()
words=[]
classes=[]
pattern_word_tags_list=[]
ignore_words=['?','!',',','.',"'s","'m"]
train_data_file=open('intents.json').read()
intents=json.loads(train_data_file)
def get_stem_words(words,ignore_words):
    stem_words=[]
    for word in words:
        if word not in ignore_words:
            w=stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words
def create_bot_corpus(words,classses,patter_word_tags_list,ignore_words):
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            pattern_word = nltk.word_tokenize(pattern)
            words.extend(pattern_word)
            pattern_word_tags_list.append(pattern_word, intent['tag'])
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            stem_words = get_stem_words(words, ignore_words)
    stem_words = get_stem_words(words, ignore_words)

    # Remove duplicate words from stem_words

    # sort the stem_words list and classes list

    # print stem_words
    print('stem_words list : ', stem_words)

    return stem_words, classes, pattern_word_tags_list


# Training Dataset:
# Input Text----> as Bag of Words
# Tags-----------> as Label

def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    bag = []
    for word_tags in pattern_word_tags_list:
        # example: word_tags = (['hi', 'there'], 'greetings']

        pattern_words = word_tags[0]  # ['Hi' , 'There]
        bag_of_words = []

        # stemming pattern words before creating Bag of words
        stemmed_pattern_word = get_stem_words(pattern_words, ignore_words)

        # Input data encoding
        '''
        Write BOW algo :
        1) take a word from stem_words list
        2) check if that word is in stemmed_pattern_word
        3) append 1 in BOW, otherwise append 0
        '''

        bag.append(bag_of_words)

    return np.array(bag)


def class_label_encoding(classes, pattern_word_tags_list):
    labels = []

    for word_tags in pattern_word_tags_list:
        # Start with list of 0s
        labels_encoding = list([0] * len(classes))

        # example: word_tags = (['hi', 'there'], 'greetings']

        tag = word_tags[1]  # 'greetings'

        tag_index = classes.index(tag)

        # Labels Encoding
        labels_encoding[tag_index] = 1

        labels.append(labels_encoding)

    return np.array(labels)


def preprocess_train_data():
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)

    # Convert Stem words and Classes to Python pickel file format

    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    train_y = class_label_encoding(tag_classes, word_tags_list)

    return train_x, train_y


bow_data, label_data = preprocess_train_data()

# after completing the code, remove comment from print statements
# print("first BOW encoding: " , bow_data[0])
# print("first Label encoding: " , label_data[0])
