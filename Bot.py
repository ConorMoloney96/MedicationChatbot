import nltk

# nltk.download() # for downloading packages

import numpy as np
import sklearn
import random
import string # to process standard python strings


f=open('nutrition.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()# converts to lowercase
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

#slices the sent tokens up
sent_tokens[:2]


word_tokens[:5]

#Stemming is a process in NLP where a word is reduced to it's stem or base format
#i.e. stems, stemming and stemmed would be reduced to stem
#Lematization is a variant of stemming where the created word must be valid
#Stemming may create non-existant words whereas lematization only creates valid words
#i.e. run is the base for of running or ran
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey", "how are you?", "howdy")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me", "howdy partner"]



# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

#Converts documents into a matrix of TF-IDF features
#TF-IDF (Term Frequency - Inverse Document Frequency) is a modification of the bag of words technique 
#Words which appear frequently in multiple documents (such as "the") are penalized
#Term Frequency is how often a term appears in a document
#Combined with cosine-similarity this allows us to find the similarity between words in the corpus and words entered by the user
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Generating response
#Takes as input user's response/input
def response(user_response):
    robo_response=''
	#implements term-frequency inverse document frequency
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
	#tf-idf is applied to all sentence tokens
    tfidf = TfidfVec.fit_transform(sent_tokens)
	#tf-idf is a transformation applied to a text to get 2 real-valued vectors in vector space
	#We then get cosine similarity to determine the similarity between vectors in this case representing the input of the user and the corpus
    #gets cosine similarity between all sentence tokens (the corpus) and last token (user input)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I'm sorry, I didn't understand. Please try again"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


flag=True
print("ROBOTo: Hello. I am a bot. I will answer your questions about nutrition")

while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBOTO: You're welcome.")
        else:
		    #if user input is recognized as a greeting
            if(greeting(user_response)!=None):
                print("ROBOTO: "+greeting(user_response))
            else:
			    #the users input/response is added to the list of tokens retrieved from the file
                sent_tokens.append(user_response)
                word_tokens=word_tokens+nltk.word_tokenize(user_response)
                final_words=list(set(word_tokens))
                print("ROBOTO: ",end="")
			    #generates a response to the user input i.e. identifies the relevant information
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBOTO: Goodbye and best of luck.")    
        
