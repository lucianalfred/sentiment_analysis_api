import pandas as pd
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report

# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Criar instância do analisador de sentimentos ANTES da função que o usa
analyzer = SentimentIntensityAnalyzer()

# Função de pré-processamento
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)

# Função para obter sentimento
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 1 if scores['pos'] > 0 else 0
    return sentiment

# Carregar dataset
df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')

# Aplicar pré-processamento
df['reviewText'] = df['reviewText'].astype(str).apply(preprocess_text)

# Aplicar análise de sentimentos
df['sentiment'] = df['reviewText'].apply(get_sentiment)

# Avaliar o desempenho
print(confusion_matrix(df['Positive'], df['sentiment']))
print(classification_report(df['Positive'], df['sentiment']))
