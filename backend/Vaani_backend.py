import tensorflow as tf
import numpy as np
import pandas as pd
from threading import Thread
import spacy
nlp = spacy.load('en_core_web_lg')

MAX_LENGTH = 256

class VaaniPredictor:
    """
    The Brains of Vaani. Loads all three encoded datasets and the model that we will be using.
    Initial versions do not have a text generator/transformer, because we did not have the 
    compute power necessary to create one.

    Use the `predict` function to send in a string query and get the output dictionary.
    """

    def __init__(self) -> None:
        bhagvadgita = pd.read_csv('../dataset/bhagvadgita_encoded.csv')
        bhagvadgita = bhagvadgita.drop(['Unnamed: 0'], axis=1)
        bhagvadgita = bhagvadgita.drop([546], axis=0) # removing this verse entirely because of very weird artifacts.
        quran = pd.read_csv('../dataset/quran_encoded.csv')
        quran = quran.drop(['Unnamed: 0'], axis=1)
        bible = pd.read_csv('../dataset/bible_encoded.csv')
        bible = bible.drop(['Unnamed: 0'], axis=1)
        model = tf.keras.models.load_model('./models/BEST_max_pooled_autoencoder.h5')
        encoder = tf.keras.Model(model.input, model.layers[5].output)
    
    def predict(self, query: str) -> list:
        """
        This function takes a string query and returns a 
        nested dictionary that can directly be converted to JSON and sent to the frontend.
        """

