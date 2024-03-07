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

# We create a 'lemmatizer' that helps us get the root form of words. 
# A lemmatizer is a tool that changes words into their base or root form. For example, it changes "running" into "run"
lemmatizer = WordNetLemmatizer()  # Create a tool to turn words into their base form.

# Open the 'intents.json' file and read the content into a Python dictionary.
intents = json.loads(open('intents.json').read())

# Lists to hold words, categories (classes), and pairs of patterns and categories.
words = []  # This will hold all individual words from patterns.
classes = []  # This will hold all the types of things users might want to do, like asking for help or buying something.
documents = []  # This will pair up patterns with intents.
ignoredLetters = ['?', '!', '.', ',']  # These are characters we don't care about.

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
            classes.append(intent['tag']) # If the tag is not in classes, it gets added. classes will eventually contain all the unique tags


# Now we make sure each word is in its root form and our lists only have unique items.
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoredLetters]  # Lemmatize words not in ignoredLetters.
words = sorted(set(word))  # Sort the words and remove any duplicates.

classes = sorted(set(classes))  # Sort the classes and remove any duplicates.