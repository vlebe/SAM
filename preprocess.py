import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

data_df = pd.read_csv('data.csv')
labels_df = pd.read_csv('labels.csv')
txt_data_df = pd.read_csv('txt_data.csv')
merged_df = pd.merge(data_df, labels_df, on=['speaker', 'dyad']) 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    
    text = text.lower()
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('french'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


merged_df['tokens'] = merged_df['text'].apply(tokenize_text)

word2vec_model = Word2Vec(merged_df['tokens'], vector_size=100, window=5, min_count=1, workers=4)

def document_vector(doc):
    doc = [word for word in doc if word in word2vec_model.wv.index_to_key]
    return np.mean(word2vec_model.wv[doc], axis=0)

merged_df['doc_vector'] = merged_df['tokens'].apply(document_vector)

vector_df = pd.DataFrame(merged_df['doc_vector'].tolist())

vector_df['turn_after'] = merged_df['turn_after']

vector_df.to_csv('word2vec_data.csv', index=False)

