# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Load ACRS dataset
acrs_df = pd.read_csv('ACRS.csv')  # Assuming ACRS.csv is in the same directory as this script

# Train sentiment analysis model
def train_sentiment_analysis_model(acrs_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(acrs_df['review'])
    clf = LogisticRegression()
    clf.fit(X, acrs_df['sentiment_score'])
    return clf, vectorizer

# Predict sentiment using the trained model
def predict_sentiment(clf, vectorizer, review):
    review_vector = vectorizer.transform([review])
    sentiment_prediction = clf.predict(review_vector)
    return sentiment_prediction

# Build recommendation system
def build_recommendation_system(acrs_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(acrs_df['review'])
    knn = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
    knn.fit(X)
    return knn, vectorizer

# Recommend products based on user's review
def recommend_products(knn, vectorizer, user_review):
    user_review_vector = vectorizer.transform([user_review])
    _, indices = knn.kneighbors(user_review_vector, n_neighbors=10)
    recommended_products = acrs_df.loc[indices[0]]['product_id'].tolist()
    return recommended_products

# Example usage:
# sentiment_clf, sentiment_vectorizer = train_sentiment_analysis_model(acrs_df)
# recommendation_knn, recommendation_vectorizer = build_recommendation_system(acrs_df)
# user_review = "This product is amazing!"
# sentiment = predict_sentiment(sentiment_clf, sentiment_vectorizer, user_review)
# recommended_products = recommend_products(recommendation_knn, recommendation_vectorizer, user_review)
