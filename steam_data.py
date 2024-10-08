import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import plotly.express as px
from transformers import pipeline
from tqdm import tqdm
#Vader
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#Roberta
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax


np.random.seed(42)

def load_data(df_name = 'steam_reviews.csv', sample_size = 0.1):
    total_rows = sum(1 for row in open('data/' + df_name)) - 1
    sample_size = int(total_rows * sample_size)

    steam_reviews = pd.read_csv('data/' + df_name, skiprows=lambda x: x > 0 and np.random.rand() > 0.1, nrows=sample_size)

    return steam_reviews

def get_english_reviews(steam_reviews):
    # Filter out non-english reviews
    steam_reviews_english = steam_reviews[(steam_reviews["language"] == "english")]
    steam_reviews_english = steam_reviews_english[['app_id', 'app_name', 'review', 'review_id','timestamp_created', 'timestamp_updated', 'recommended', 'author.steamid', 'weighted_vote_score']]
    steam_reviews_english = steam_reviews_english.sort_values(by=['timestamp_created', 'timestamp_updated'], ascending=[True, True])

    # Replace the author.steamid with smaller numbers starting from 0
    steam_reviews_english['author.steamid'] = pd.factorize(steam_reviews_english['author.steamid'])[0]

    # Replace the app_id with smaller numbers starting from 0
    steam_reviews_english['app_id'] = pd.factorize(steam_reviews_english['app_id'])[0]

    # Replace the review_id with smaller numbers starting from 0
    steam_reviews_english['review_id'] = pd.factorize(steam_reviews_english['review_id'])[0]

    return steam_reviews_english

def get_updated_steam_reviews(steam_reviews):
    # Reviews that were updated
    steam_reviews_updated = steam_reviews[(steam_reviews['timestamp_created'] != steam_reviews['timestamp_updated'])]

    return steam_reviews_updated


def get_not_updated_steam_reviews(steam_reviews):
    # Reviews that were updated
    steam_reviews_not_updated = steam_reviews[(steam_reviews['timestamp_created'] == steam_reviews['timestamp_updated'])]

    return steam_reviews_not_updated


def create_review_df(df):
    data = []
    for _, row in df.iterrows():
        app_id = row['app_id']
        app_name = row['app_name']
        review_id = row['review_id']
        review = row['review']
        timestamp = row['timestamp_created']
        timestamp_updated = row['timestamp_updated']
        recommended = row['recommended']
        author_id = row['author.steamid']
        weighted_vote_score = row['weighted_vote_score']
        data.append([app_id, app_name, review, review_id,timestamp, recommended, author_id, weighted_vote_score])
        # Add the updated review if it exists
        if timestamp != timestamp_updated:
            data.append([app_id, app_name, review, review_id,timestamp_updated, recommended, author_id, weighted_vote_score])
    
    new_df = pd.DataFrame(data, columns=['app_id', 'app_name', 'review', 'review_id', 
                                         'timestamp', 'recommended', 'author_id', 
                                         'weighted_vote_score'])
    new_df = new_df.sort_values(by=['timestamp'], ascending=[True])
    return new_df

def merge_and_order_reviews(df1, df2, parameter = 'timestamp'):
    df_ordered = pd.concat([df1, df2]).sort_values(by=[parameter], ascending=[True])
    df_ordered['review'] = df_ordered['review'].fillna('', inplace=True)

    return df_ordered


# Roberta Model
def preprocess_text(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def get_sentiment_scores_roberta(df, text_column_name='review'):
    # Load the model
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    sentiment_scores = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing reviews"):
        review = row[text_column_name]
        review = preprocess_text(review)
        encoded_input = tokenizer(review, return_tensors='pt', padding=True, truncation=True, max_length=512)
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        sentiment_scores.append(scores)
    return sentiment_scores


def add_sentiment_scores_to_df(df, sentiment_scores):
    # Add the sentiment scores to the DataFrame
    sentiment_columns = ['neg', 'neu', 'pos']
    steam_reviews_all_roberta = df.copy()
    steam_reviews_all_roberta[sentiment_columns] = pd.DataFrame(sentiment_scores, index=df.index)

    return steam_reviews_all_roberta


def main():
    # Load data
    steam_reviews = load_data()

    # Get English reviews
    steam_reviews_english = get_english_reviews(steam_reviews)

    # Get updated and not updated reviews
    steam_reviews_updated = get_updated_steam_reviews(steam_reviews_english)
    steam_reviews_not_updated = get_not_updated_steam_reviews(steam_reviews_english)

    # Create review DataFrame
    steam_reviews_updated = create_review_df(steam_reviews_updated)
    steam_reviews_not_updated = create_review_df(steam_reviews_not_updated)

    # Merge and order reviews
    steam_reviews_all = merge_and_order_reviews(steam_reviews_updated, steam_reviews_not_updated)


    # Get sentiment scores using Roberta model
    sentiment_scores = get_sentiment_scores_roberta(steam_reviews_all)

    # Add sentiment scores to DataFrame
    steam_reviews_all_with_sentiment = add_sentiment_scores_to_df(steam_reviews_all, sentiment_scores)

    # Save the final DataFrame to a CSV file
    steam_reviews_all_with_sentiment.to_csv('data/steam_reviews_roberta.csv', index=False)

if __name__ == "__main__":
    main()

