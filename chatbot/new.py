# This Python file sets up the basic components needed for a chatbot.
# It prepares the data from 'intents.json' to understand and respond to user messages.
# The chatbot will learn from examples provided in 'intents.json'.

import random  # Used to generate random responses.
import json  # Helps with reading and writing json files.
import pickle  # Used to save and load Python objects to and from files.
import numpy as np  # Helpful for numerical operations.
import tensorflow as tf  # Provides tools for machine learning.

import nltk  # Natural Language Toolkit for processing human language.
from nltk.stem import WordNetLemmatizer  # Helps get the root form of words.

lemmatizer = WordNetLemmatizer()  # Create a tool to turn words into their base form.

# Open the 'intents.json' file and read the content into a Python dictionary.
intents = json.loads(open('intents.json').read())

# Lists to hold words, categories (classes), and pairs of patterns and categories.
words = []
classes = []
documents = []
ignoredLetters = ['?', '!', '.', ',']  # Punctuation marks to ignore.

# Go through each intent and process the patterns and tags.
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Break each pattern into words.
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)  # Add the words to our words list.
        # Add a tuple of words and the tag to our documents list.
        documents.append((wordList, intent['tag']))
        # If the tag is not already in our classes list, add it.
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize the words and remove duplicates.
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoredLetters]
words = sorted(set(words))  # Sort the words.

classes = sorted(set(classes))  # Remove duplicates and sort the classes.
