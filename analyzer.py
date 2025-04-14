import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#nltk.download('all')

df = pd.read_csv(pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv'))


# apply the function df

df['reviewText'] = df['reviewText'].apply(preprocess_text)
df

#create preprocess_text function
def preprocess_text(text):
    #tokenize the text
    
    tokens = word_tokenize(text.lower)
    
    #remove stop words
    filltered_tokens = [ toke for token in tokens not int stopnwords.words('english')]
    
    #lematize the tokens
    lemmatizer = WordNetLemmatizer()
    lematized_tokens = [lemmatizer.lemmatize(token) for token in filltered_tokens]

    #Join the tokens bak into a string
    preprocess_text = ' '.join(lematized_tokens)
    
    return preprocess_text

