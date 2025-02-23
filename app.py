import nltk
import string
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fastembed
nltk.download('punkt')
nltk.download('stopwords')


# Data loading and preprocessing
df = pd.read_csv('data.csv')
df = df.dropna(subset=['title', 'generes','description'])
df = df.sort_values('title').assign(genre_len=df['generes'].str.len()).sort_values(['title', 'genre_len'], ascending=[True, False])
df = df.drop_duplicates('title', keep='first').drop('genre_len', axis=1)
df = df.drop(['Unnamed: 0'], axis=1)
df['combined_text'] = (df['title'] + " " + df['author'] + " " + df['generes'] + " " + df['description'])


# Text preprocessing function
def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))

    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    return " ".join([word for word in tokens if word not in stop_words])


# Embedding Model Class
class embeddingModel:
    def __init__(self, model_type="tfidf", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_type = model_type
        if model_type == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.vectorizer.fit_transform(df["combined_text"])
        elif model_type == "fastembed":
            self.model = fastembed.TextEmbedding(model_name=model_name)
            
    def embed_documents(self, docs):
        if self.model_type == "fastembed":
            return [list(self.model.embed(doc))[0] for doc in docs]
        elif self.model_type == "tfidf":
            return [list(vec) for vec in self.tfidf_matrix.toarray()]
            
    def embed_query(self, query):
        if self.model_type == "fastembed":
            return list(self.model.embed(query))[0]
        elif self.model_type == "tfidf":
            return self.vectorizer.transform([query]).toarray()[0]
        

# Similarity Ranking Function
def similarity_ranking(query_emb, doc_embeddings, n_recommendations=5):
    similarities = [cosine_similarity([query_emb], [doc_emb])[0][0] for doc_emb in doc_embeddings]
    top_indices = np.argsort(similarities)[-n_recommendations:][::-1]
    return top_indices, [similarities[i] for i in top_indices]


def main():
    while True:
        user_query = input("\nEnter your query or type 'exit' to close the app: ").strip()
        if user_query.lower() == "exit":
            break

        model_choice = input("Choose model (tfidf/fastembed): ").strip().lower()
        if model_choice not in ["tfidf", "fastembed"]:
            model_choice = "tfidf"

        embedder = embeddingModel(model_type=model_choice)
        # Preprocess query and generate embeddings
        query_processed = preprocess_text(user_query)
        query_emb = embedder.embed_query(query_processed)

        # Preprocess documents and generate embeddings  
        doc_processed = df['combined_text'].apply(preprocess_text)
        doc_emb = embedder.embed_documents(doc_processed)

        # Compute similarity and get top recommendations
        top_indices, top_similarities = similarity_ranking(query_emb, doc_emb, n_recommendations=5)

        # Display recommendations
        for idx, sim_score in zip(top_indices, top_similarities):
            row = df.iloc[idx]
            print(f"- {row['title']} by {row['author']} | Genre: {row['generes']} | Similarity Score: {sim_score:.4f}")


# Run the Program
if __name__ == '__main__':
    main()