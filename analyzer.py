import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('all')

df = pd.read_csv(pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv'))

df