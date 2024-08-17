from django.http import HttpResponse 
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf
# import keras
import numpy as np
import pandas as pd
# import tensorflow.compat.v1 as tf
# future updates for multithreading


import spacy
# Check if the 'en_core_web_lg' model is installed
try:
    nlp = spacy.load('en_core_web_lg')
except OSError:
    print("Model 'en_core_web_lg' not found. Installing now...")
    spacy.cli.download('en_core_web_lg')
    nlp = spacy.load('en_core_web_lg')
    print("Model 'en_core_web_lg' installed successfully.")

MAX_LENGTH = 256

class VaaniPredictor:
    

    def __init__(self) -> None:
        bhagvadgita = pd.read_csv('../dataset/bhagvadgita_encoded.csv')
        bhagvadgita = bhagvadgita.drop(['Unnamed: 0'], axis=1)
        self.bhagvadgita = bhagvadgita.drop([546], axis=0) 
        quran = pd.read_csv('../dataset/quran_encoded.csv')
        self.quran = quran.drop(['Unnamed: 0'], axis=1)
        bible = pd.read_csv('../dataset/bible_encoded.csv')
        self.bible = bible.drop(['Unnamed: 0'], axis=1)
        model = tf.keras.models.load_model('../backend/models/BEST_max_pooled_autoencoder.h5')
        self.encoder = tf.keras.Model(model.input, model.layers[5].output)
        self.vectorizer = spacy.load('en_core_web_lg')
    
    def get_tensor(self, query: str) -> tuple:
        
        maxtokens = []
        meantokens = []
        alltokens = []

        doc = nlp(query)
        tokenlist = np.array([token.vector for token in doc])
        maxnormlist = np.array([float(token.vector_norm) for token in doc])
        if MAX_LENGTH - tokenlist.shape[0] > 0:
            constructarr = np.zeros((MAX_LENGTH - tokenlist.shape[0], 300))
            alltokens.append(np.append(tokenlist, constructarr, axis=(0)))
        else:
            alltokens.append(tokenlist[:MAX_LENGTH])

        maxtokens.append(np.array(tokenlist[maxnormlist.argmax()]))
        meantokens.append(np.array(tokenlist.mean(axis=(0))))

        tmaxtokens = tf.convert_to_tensor(maxtokens)
        tmeantokens = tf.convert_to_tensor(meantokens)
        talltokens = tf.convert_to_tensor(alltokens)

        return (tmaxtokens, tmeantokens, talltokens)

    def find_closest_gita_verse(self, token: np.ndarray) -> dict:
        
        vectorComp = self.bhagvadgita.iloc[:, -5:].to_numpy()
        diff = np.array([np.dot(token, i)/(np.linalg.norm(token)*np.linalg.norm(i)) for i in vectorComp])
        seriesSlice = self.bhagvadgita.iloc[diff.argmax(), 1:4]
        return { 'chapter' : seriesSlice[0], 'verse' : seriesSlice[1], 'text' : seriesSlice[2] }
    
    def find_closest_bible_verse(self, token: np.ndarray) -> dict:
        
        vectorComp = self.bible.iloc[:, -5:].to_numpy()
        diff = np.array([np.dot(token, i)/(np.linalg.norm(token)*np.linalg.norm(i)) for i in vectorComp])
        seriesSlice = self.bible.iloc[diff.argmax(), 1:5]
        return { 'book' : seriesSlice[0],'chapter' : seriesSlice[1], 'verse' : seriesSlice[2], 'text' : seriesSlice[3] }
    
    def find_closest_quran_verse(self, token: np.ndarray) -> dict:
        
        vectorComp = self.quran.iloc[:, -5:].to_numpy()
        diff = np.array([np.dot(token, i)/(np.linalg.norm(token)*np.linalg.norm(i)) for i in vectorComp])
        seriesSlice = self.quran.iloc[diff.argmax(), 1:5]
        return { 'surah' : seriesSlice[0], 'ayat' : seriesSlice[1], 'verse' : seriesSlice[2], 'tafseer' : seriesSlice[3] }

    
    def predict(self, query: str) -> dict:
        """
        This function takes a string query and returns a 
        nested dictionary that can directly be converted to JSON and sent to the frontend.
        """
        max_query_vector, mean_query_vector, _ = self.get_tensor(query)
        max_encoding = self.encoder.predict(max_query_vector, verbose=0)
        mean_encoding = self.encoder.predict(mean_query_vector, verbose=0)
        biblemax = self.find_closest_bible_verse(max_encoding)
        biblemean = self.find_closest_bible_verse(mean_encoding)
        gitamax = self.find_closest_gita_verse(max_encoding)
        gitamean = self.find_closest_gita_verse(mean_encoding)
        quranmax = self.find_closest_quran_verse(max_encoding)
        quranmean = self.find_closest_quran_verse(mean_encoding)
        result = {'bible' : [biblemax, biblemean], 'bhagvadgita' : [gitamax, gitamean], 'quran' : [quranmax, quranmean]}
        return result

# if __name__ == '__main__':
#     from pprint import pprint
#     print("Welcome to Vaani's Backend!\n---------------------------------\n")
#     predictor = VaaniPredictor()
#     query = input("What troubles you, friend?\n=> ")
#     # pprint(predictor.predict(query))
# class MyForm(forms.Form):
#     my_text = forms.CharField(widget=forms.Textarea)
# @api_view(['GET', 'POST'])

@csrf_exempt
def home(request):

    # print("Welcome to Vaani's Backend!\n---------------------------------\n")
    predictor = VaaniPredictor()
    
    if request.method == 'POST':
        query = request.POST.get('input1')
        answer = (predictor.predict(query))
        ans1 = answer['bhagvadgita'][0]['text'] + "\n" + answer['bhagvadgita'][1]['text']
        ans2 = answer['bible'][0]['text'] + "\n" + answer['bible'][1]['text']
        ans3 = answer['quran'][0]['verse'] + "\n" + answer['quran'][1]['verse']
        return render(request, "home.html", {'answer1' : ans1, 'answer2' : ans2, 'answer3' : ans3})
    else:
        ans4 = ""
        ans5 = ""
        ans6 = ""
        return render(request, "home.html", {'answer1' : ans4, 'answer2' : ans5, 'answer3' : ans6})

    
    
    


"""1
Sample input: What is the meaning of life?
Sample output:
{
    'bhagvadgita': [
        {
            'chapter': 4,
            'text': 'Some Yogis sacrifice all the functions of their ''senses ans all their vital functions of life, into ''the fire of Yoga, in the shape of self control, ''which is kindled by Gyan (wisdom). (This signifies ''the sacrificer?s one-mindedness with the Lord, the ''Supreme Goal).',
            'verse': 27
        },
        {
            'chapter': 9,
            'text': 'O Arjuna, this world is one that is quickly ''passing, very brief and full of sufferings. Having ''been born here in such a world, the only way that ''one can attain true happiness and peace is to ''worship Me.',
            'verse': 33
        }
    ],
    'bible': [
        {
            'book': 10,
            'chapter': 1,
            'text': '(It is recorded in the book of Jashar for teaching to the ''sons of Judah) and he said:',
            'verse': 18
        },
        {
            'book': 43,
            'chapter': 12,
            'text': 'Have no fear, daughter of Zion: see your King is coming, ''seated on a young ass.',
            'verse': 15
        }
    ],
    'quran': [
        {
            'ayat': 11,
            'surah': 6,
            'tafseer': 'Say to them ‘Travel in the land and see the nature of ''the consequence for the deniers’ of the messengers how ''they were destroyed through chastisement; perhaps they ''will take heed.',
            'verse': 'Say (unto the disbelievers): Travel in the land, and see ''the nature of the consequence for the rejecters!'
        },
        {
            'ayat': 1,
            'surah': 49,
            'tafseer': 'O you who believe do not venture ahead of tuqaddimū ''derives from qaddama with the sense of the 5th form ''taqaddama that is to say do not come forward with any ''unwarranted saying or deed ahead of God and His ''Messenger the one communicating the Message from Him ''that is to say without their permission and fear God. ''Surely God is Hearer of your sayings Knower of your ''deeds this was revealed regarding the dispute between ''Abū Bakr and ‘Umar may God be pleased with them both ''in the presence of the Prophet s over the appointment ''of al-Aqra‘ b. Hābis or al-Qa‘qā‘ b. Ma‘bad as ''commander of his tribe.',
            'verse': 'O ye who believe! Be not forward in the presence of ''Allah and His messenger, and keep your duty to Allah. ''Lo! Allah is Hearer, Knower.'
        }
    ]
}
"""



# Create your views here.
