# We bring in some code from other files that we need for our chatbot to work.
import random
import json
import pickle
import numpy as np
import tensorflow as tf

# nltk is a toolkit we use to work with language data.
import nltk
from nltk.stem import WordNetLemmatizer

# We create a 'lemmatizer' that helps us get the root form of words. 
# A lemmatizer is a tool that changes words into their base or root form. For example, it changes "running" into "run"
lemmatizer = WordNetLemmatizer()

# We load our chatbot's 'intents' from a file which tells it how to reply to different things people might say.
intents = json.loads(open('intents.json').read())

# Here we set up some lists to store different parts of chatbot's language understanding.
words = []  # This will hold all individual words from patterns.
classes = []  # This will hold all the types of things users might want to do, like asking for help or buying something.
documents = []  # This will pair up patterns with intents.
ignoredLetters = ['?', '!', '.', ',']  # These are characters we don't care about.

# We go through all our intents and each pattern in them to fill our lists above.
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # We break the pattern into individual words.
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)  # Add the words from this pattern to our big list of words.
        documents.append((wordList, intent['tag']))  #  creates a tuple of the tokenized pattern and its associated tag, then adds it to the documents list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])  # If the tag is not in classes, it gets added. classes will eventually contain all the unique tags 

# Now we make sure each word is in its root form and our lists only have unique items.
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoredLetters]  # Lemmatize words not in ignoredLetters.
words = sorted(set(word))  # Sort the words and remove any duplicates.

classes = sorted(set(classes))  # Sort the classes and remove any duplicates.
