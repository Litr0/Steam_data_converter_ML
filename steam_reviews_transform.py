# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import plotly.express as px
from transformers import pipeline
from tqdm import tqdm

# %%
np.random.seed(42)

# %% [markdown]
# ### Exploring the data
# Data source: https://www.kaggle.com/datasets/najzeko/steam-reviews-2021/data

# %%
# Get the total number of rows in the CSV file
total_rows = sum(1 for row in open('data/steam_reviews.csv')) - 1  # subtract 1 for header
sample_size = int(total_rows * 0.1)

# Read only 10% of the CSV file
steam_reviews = pd.read_csv('data/steam_reviews.csv', skiprows=lambda x: x > 0 and np.random.rand() > 0.1, nrows=sample_size)

# %%
# Get all unique app names present in the dataset
app_names = steam_reviews['app_name'].unique()
app_names = app_names.tolist()
app_names

# %%
# Number of reviews in English
steam_reviews[(steam_reviews["language"] == "english")].shape

# %%
#Possible review Bombing for GTA V between 2017-06-01 and 2017-07-31
one_game_only_english = steam_reviews[(steam_reviews["app_name"].str.contains("Grand Theft Auto", case = False)) 
                                      & (steam_reviews["language"] == "english")
                                      & (steam_reviews["recommended"] == False)
                                      & (steam_reviews["timestamp_updated"] > 1496268000)
                                      & (steam_reviews["timestamp_updated"] < 1501538399)
                                      & ((steam_reviews["review"].str.contains("Take-Two", case = False)) 
                                         | (steam_reviews["review"].str.contains("OpenIV", case = False)))
                                      & (steam_reviews["votes_funny"] == 0)]
one_game_only_english.head()

# %%
total_gta_reviews = steam_reviews[(steam_reviews["app_name"].str.contains("Grand Theft Auto", case = False))
                                  & (steam_reviews["language"] == "english")
                                  & (steam_reviews["timestamp_updated"] > 1496268000)
                                  & (steam_reviews["timestamp_updated"] < 1501538399)]
total_gta_reviews.head()

# %%
# Compare the number of rows of both tables
rows_one_game_only_english = one_game_only_english.shape[0]
rows_total_gta_reviews = total_gta_reviews.shape[0]

print(f"Number of rows in one_game_only_english: {rows_one_game_only_english}")
print(f"Number of rows in total_gta_reviews: {rows_total_gta_reviews}")
print(f"Percentage of review bombing: {rows_one_game_only_english / rows_total_gta_reviews * 100:.2f}%")

# %% [markdown]
# ### Setting the basis for the graph to build

# %%
steam_reviews_english = steam_reviews[(steam_reviews["language"] == "english")]
steam_reviews_english = steam_reviews_english[['app_id', 'app_name', 'review', 'review_id','timestamp_created', 'timestamp_updated', 'recommended', 'author.steamid', 'weighted_vote_score']]
steam_reviews_english = steam_reviews_english.sort_values(by=['timestamp_created', 'timestamp_updated'], ascending=[True, True])

# Replace the author.steamid with smaller numbers starting from 0
steam_reviews_english['author.steamid'] = pd.factorize(steam_reviews_english['author.steamid'])[0]

# Replace the app_id with smaller numbers starting from 0
steam_reviews_english['app_id'] = pd.factorize(steam_reviews_english['app_id'])[0]

# Replace the review_id with smaller numbers starting from 0

steam_reviews_english['review_id'] = pd.factorize(steam_reviews_english['review_id'])[0]

steam_reviews_english.head()

# %%
# Reviews that were updated
steam_reviews_updated = steam_reviews_english[(steam_reviews_english['timestamp_created'] != steam_reviews_english['timestamp_updated'])]

# Reviews that were not updated
steam_reviews_not_updated = steam_reviews_english[(steam_reviews_english['timestamp_created'] == steam_reviews_english['timestamp_updated'])]

# %%
steam_reviews_updated.head()

# %%
steam_reviews_not_updated.head()

# %%
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

# %%
def merge_and_order_reviews(df1, df2, parameter = 'timestamp'):
    return pd.concat([df1, df2]).sort_values(by=[parameter], ascending=[True])

# %%
steam_reviews_updated = create_review_df(steam_reviews_updated)
steam_reviews_not_updated = create_review_df(steam_reviews_not_updated)
steam_reviews_all = merge_and_order_reviews(steam_reviews_updated, steam_reviews_not_updated)

# %%
steam_reviews_all.head()

# %%
steam_reviews_all['review'].fillna('', inplace=True)

# %% [markdown]
# ### Parse the reviews as LIWC feature vector usign VADER

# %%
import nltk

nltk.download('vader_lexicon')

# %%
# Bad review of GTA V
review = steam_reviews_all[(steam_reviews_all['author_id'] == 67820)
                           & (steam_reviews_all['timestamp'] == 1428975119)]['review'].values[0]
review

# %%
# Test the sentiment analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create an instance of the Vader sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# List of example texts to analyze
texts = [
    review 
]

# Loop through the texts and get the sentiment scores for each one
for text in texts:
    scores = analyzer.polarity_scores(text)
    print(text)
    print(type(scores), scores)

# %%
# Function to get the sentiment score all reviews in the dataset
def get_sentiment_scores(df, text_column_name='review'):
    # Create an instance of the Vader sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # List to store the sentiment scores
    sentiment_scores = []
    
    # Loop through all the reviews
    for _, row in df.iterrows():
        # Get the review text
        review = row[text_column_name]

        if type(review) != str:
            print("Review is Not a String")
            print(row)
        # Get the sentiment scores
        scores = analyzer.polarity_scores(review)

        # Transform the scores into a array
        scores = [scores['neg'], scores['neu'], scores['pos'], scores['compound']]

        # Append the scores to the list
        sentiment_scores.append(scores)
    
    return sentiment_scores

# %%
sentiment_scores_steam_reviews = get_sentiment_scores(steam_reviews_all)
sentiment_scores_steam_reviews[:5]

# %%
# Add the sentiment scores to the DataFrame
sentiment_columns = ['neg', 'neu', 'pos', 'compound']
steam_reviews_all_vader = steam_reviews_all.copy()
steam_reviews_all_vader[sentiment_columns] = pd.DataFrame(sentiment_scores_steam_reviews, index=steam_reviews_all.index)
steam_reviews_all_vader.head()

# %% [markdown]
# ### Attempt to vectorize using Roberta Pre-trained Model
# Source: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

# %%
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax

# %%
def preprocess_text(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)

# %%
# Testing the model
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

model = AutoModelForSequenceClassification.from_pretrained(MODEL)

text = review

text = preprocess_text(text)
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)

ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = model.config.id2label[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i + 1}) {l} {np.round(float(s), 4)}")

# %%
def get_sentiment_scores_roberta(df, text_column_name='review'):
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

# %%
# Ensure the tokenizer and model are using the same configuration and vocabulary
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

model = AutoModelForSequenceClassification.from_pretrained(MODEL)

sentiment_scores_steam_reviews_roberta = get_sentiment_scores_roberta(steam_reviews_all)
sentiment_scores_steam_reviews_roberta[:5]

# %%
# Add the sentiment scores to the DataFrame
sentiment_columns = ['neg', 'neu', 'pos']
steam_reviews_all_roberta = steam_reviews_all.copy()
steam_reviews_all_roberta[sentiment_columns] = pd.DataFrame(sentiment_scores_steam_reviews_roberta, index=steam_reviews_all.index)
steam_reviews_all_roberta.head()

# %% [markdown]
# ### The network should be in the following format:
# 
# - One line per interaction/edge.
# - Each line should be: user, item, timestamp, state label, comma-separated array of features.
# - First line is the network format.
# - User and item fields can be alphanumeric.
# - Timestamp should be in cardinal format (not in datetime).
# - State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
# - Feature list can be as long as desired. It should be atleast 1 dimensional. If there are no features, use 0 for all interactions.
# 

# %%
def transform_to_network(df):
    network_data = []
    for _, row in df.iterrows():
        user = row['author.steamid']
        item = row['app_id']
        timestamp = row['timestamp']
        state_label = 0
        negative = row['neg']
        neutral = row['neu']
        positive = row['pos']
        # Add the features list to the network data
        array_to_append = [user, item, timestamp, state_label, negative, neutral, positive]
        network_data.append(array_to_append)
    # Create a DataFrame from the network data
    network_df = pd.DataFrame(network_data, columns=['user_id', 'item_id', 'timestamp', 'state_label', 'negative', 'neutral', 'positive'])
    return network_df

# %%
# Transform the steam_reviews in english DataFrame
network_df = transform_to_network(steam_reviews_english)

# Display the first few rows of the transformed network DataFrame
network_df.head()

# %%
network_df.rename(columns={'negative': 'comma_separated_list_of_features', 'neutral': '', 'positive': ''}, inplace=True)
network_df.head()

# %%
# Save the network DataFrame to a CSV file
network_df.to_csv('data/steam.csv', index=False)
