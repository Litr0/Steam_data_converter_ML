import re
from networkx import is_empty
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import plotly.express as px
import test
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
import zipfile
#Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples
import time
import hdbscan  # type: ignore
#Results analysis
import pickle
# PhishGNN
import torch
from torch_geometric.data import Data
import os
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity




np.random.seed(42)

def load_data(df_name = 'steam_reviews.csv', sample_size = 0.1):
    total_rows = sum(1 for row in open('data/' + df_name)) - 1
    sample_size = int(total_rows * sample_size)

    print(f"Loading {sample_size} rows from {df_name}")
    steam_reviews = pd.read_csv('data/' + df_name, skiprows=lambda x: x > 0 and np.random.rand() > 0.1, nrows=sample_size)
    print(f"Loaded {steam_reviews.shape[0]} rows")

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
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Creating review DataFrame"):
        app_id = row['app_id']
        app_name = row['app_name']
        try:
            review_id = row['review_id']
        except:
            if _ == 0:
                print("No review_id column")
            review_id = 0
        review = row['review']
        timestamp = row['timestamp_created']
        timestamp_updated = row['timestamp_updated']
        recommended = row['recommended']
        try:
            author_id = row['author.steamid']
        except:
            if _ == 0:
                print("No author.steamid column")
            author_id = row['author_id']
        weighted_vote_score = row['weighted_vote_score']
        data.append([app_id, app_name, review, review_id,timestamp, recommended, author_id, weighted_vote_score])
        # Add the updated review if it exists
        if timestamp != timestamp_updated:
            data.append([app_id, app_name, review, review_id,timestamp_updated, recommended, author_id, weighted_vote_score])
    
    new_df = pd.DataFrame(data, columns=['app_id', 'app_name', 'review', 'review_id', 
                                         'timestamp', 'recommended', 'author_id', 
                                         'weighted_vote_score'])
    new_df.sort_values(by=['timestamp'], ascending=[True])
    return new_df

def merge_and_order_reviews(df1, df2, parameter = 'timestamp'):
    df_ordered = pd.concat([df1, df2]).sort_values(by=[parameter], ascending=[True])

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


def transform_to_network(df):
    network_data = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Transforming to network"):
        try:
            user = row['cluster_id']
        except:
            user = row['author_id']
        item = row['app_id']
        timestamp = row['timestamp']
        try:
            state_label = row['review_bombing']
        except:
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

def print_data_info(data):
    print("Data object information:")
    print(data)
    print("Number of nodes:", data.num_nodes)
    print("Number of edges:", data.num_edges)
    print("Number of features per node:", data.num_node_features)
    print("Number of features per edge:", data.num_edge_features)
    print("Number of classes:", data.num_classes if hasattr(data, 'num_classes') else 'N/A')
    print("Edge index:", data.edge_index)
    print("Node features:", data.x)
    print("Length of node features:", data.x.size(1) if data.x is not None else 'N/A')
    print("Edge features:", data.edge_attr if hasattr(data, 'edge_attr') else 'N/A')
    print("Labels:", data.y)

def extract_highest_probability_val(preds):
    pred_vals = []
    for pred in preds:
        if pred[0] > pred[1]:
            pred_vals.append(0)
        else:
            pred_vals.append(1)
    return pred_vals

def main():
    # Load data
    steam_reviews = pd.read_csv('data/steam_reviews.csv')

    # Get English reviews
    steam_reviews_english = get_english_reviews(steam_reviews)
    print(f"Number of English reviews: {steam_reviews_english.shape[0]}")

    # Filter out non GTA V reviews
    # steam_reviews_english = steam_reviews_english[(steam_reviews_english["app_name"].str.contains("Grand Theft Auto V", case = False))]

    # Get updated and not updated reviews
    steam_reviews_updated = get_updated_steam_reviews(steam_reviews_english)
    steam_reviews_not_updated = get_not_updated_steam_reviews(steam_reviews_english)

    # Create review DataFrame
    steam_reviews_updated = create_review_df(steam_reviews_updated)
    print(f"Number of updated reviews: {steam_reviews_updated.shape[0]}")
    steam_reviews_not_updated = create_review_df(steam_reviews_not_updated)
    print(f"Number of not updated reviews: {steam_reviews_not_updated.shape[0]}")

    # Merge and order reviews
    steam_reviews_all = merge_and_order_reviews(steam_reviews_updated, steam_reviews_not_updated)
    steam_reviews_all.fillna({'review': ''}, inplace=True)
    print(f"Number of all reviews: {steam_reviews_all.shape[0]}")

    # Filter out the reviews between January 1, 2017 and December 31, 2017
    steam_reviews_all = steam_reviews_all[(steam_reviews_all['timestamp'] >= 1483228800) & (steam_reviews_all['timestamp'] <= 1514764799)]

    print(f"Number of all reviews in 2017: {steam_reviews_all.shape[0]}")

    # Get sentiment scores using Roberta model
    sentiment_scores = get_sentiment_scores_roberta(steam_reviews_all)
    print(f"First 5 sentiment scores: {sentiment_scores[:5]}")

    # Add sentiment scores to DataFrame
    steam_reviews_all_with_sentiment = add_sentiment_scores_to_df(steam_reviews_all, sentiment_scores)

    # Save the final DataFrame to a CSV file
    steam_reviews_all_with_sentiment.to_csv('data/steam_reviews_roberta_2017.csv', index=False)


# Marking with a 1 the reviews that mention Take-Two or OpenIV in a new column called 'review_bombing'
def main_2():
    steam_reviews = pd.read_csv('data/steam_reviews_roberta_2017_new.csv')

    #Possible review Bombing for GTA V between 2017-06-01 and 2017-07-31
    gta_reviews = steam_reviews[(steam_reviews["app_name"].str.contains("Grand Theft Auto", case = False))
                                      & (steam_reviews["recommended"] == False)
                                      & (steam_reviews["timestamp"] > 1496268000)
                                      & (steam_reviews["timestamp"] < 1501538399)
                                      & ((steam_reviews["review"].str.contains("Take Two", case = False))
                                      |  (steam_reviews["review"].str.contains("Take-Two", case = False)) 
                                      |  (steam_reviews["review"].str.contains("OpenIV", case = False))
                                      |  (steam_reviews["review"].str.contains("Open IV", case = False))  
                                      |  (steam_reviews["review"].str.contains("Rockstar", case = False))
                                      |  (steam_reviews["review"].str.contains("modding", case = False))
                                      |  (steam_reviews["review"].str.contains("mod", case = False)))]
    
    firewatch_reviews = steam_reviews[(steam_reviews["app_name"].str.contains("Firewatch", case = False))
                                        & (steam_reviews["recommended"] == False)
                                        & (steam_reviews["timestamp"] >= 1504224000) 
                                        & (steam_reviews["timestamp"] <= 1506815999)
                                        & ((steam_reviews["review"].str.contains("Pewdiepie", case = False))
                                        |  (steam_reviews["review"].str.contains("Pew Die Pie", case = False)))]


    sonic_mania_reviews = steam_reviews[(steam_reviews["app_name"].str.contains("Sonic Mania", case = False))
                                        & (steam_reviews["recommended"] == False)
                                        & (steam_reviews["timestamp"] >= 1501545600)
                                        & (steam_reviews["timestamp"] <= 1504223999)  
                                        & ((steam_reviews["review"].str.contains("Denuvo", case = False))
                                        |  (steam_reviews["review"].str.contains("DRM", case = False))
                                        |  (steam_reviews["review"].str.contains("Sega", case = False)))]
    
    print(f"Number of reviews that mention Take-Two, OpenIV, modding, mod or Rockstar: {gta_reviews.shape[0]}")
    print(f"Number of reviews that mention Pewdiepie: {firewatch_reviews.shape[0]}")
    print(f"Number of reviews that mention Denuvo, DRM or Sega: {sonic_mania_reviews.shape[0]}")

    # Create a DataFrame excluding the values in one_game_only_english
    steam_reviews_excluding_bombing = steam_reviews[(~steam_reviews.index.isin(gta_reviews.index))
                                                    & (~steam_reviews.index.isin(firewatch_reviews.index))
                                                    & (~steam_reviews.index.isin(sonic_mania_reviews.index))]

    steam_reviews_excluding_bombing = steam_reviews_excluding_bombing.assign(review_bombing=0)

    gta_reviews = gta_reviews.assign(review_bombing=1)

    sonic_mania_reviews = sonic_mania_reviews.assign(review_bombing=1)

    firewatch_reviews = firewatch_reviews.assign(review_bombing=1)

    # Merge the two DataFrames
    steam_reviews_all = merge_and_order_reviews(steam_reviews_excluding_bombing, gta_reviews)

    steam_reviews_all = merge_and_order_reviews(steam_reviews_all, sonic_mania_reviews)

    steam_reviews_all = merge_and_order_reviews(steam_reviews_all, firewatch_reviews)

    steam_reviews_all.to_csv('data/steam_reviews_all_2017_new.csv', index=False)

    network_df = transform_to_network(steam_reviews_all)
    print(f"First 5 rows of the network data:\n {network_df.head()}")

    network_df.rename(columns={'negative': 'comma_separated_list_of_features', 'neutral': '', 'positive': ''}, inplace=True)

    network_df.to_csv('data/steam_2017_new.csv', index=False)

    print("Data saved to 'data/steam_2017_new.csv'")

# See how many GTA V reviews are in total
def main_3():
    steam_reviews = pd.read_csv('data/steam_reviews.csv')

    # Filter out non-english reviews
    steam_reviews_english = steam_reviews[(steam_reviews["language"] == "english")]

    # Filter out reviews for GTA V

    gta_v_reviews = steam_reviews_english[(steam_reviews_english["app_name"].str.contains("Grand Theft Auto", case = False))]

    print(f"Number of GTA V reviews: {gta_v_reviews.shape[0]}")


# See how many users and games are in the dataset
def main_4():
    steam_reviews = pd.read_csv('data/steam_reviews_all_2017_new.csv')

    # Print the size of the dataset
    print(f"Number of rows in the dataset: {steam_reviews.shape[0]}")

    # Get the number of unique users
    unique_users = steam_reviews['author_id'].nunique()
    print(f"Number of unique users: {unique_users}")

    # Get the number of unique games
    unique_games = steam_reviews['app_id'].nunique()
    print(f"Number of unique games: {unique_games}")

    # Get the number of reviews per user
    reviews_per_user = steam_reviews['author_id'].value_counts()
    print(f"Number of reviews per user:\n {reviews_per_user}")

    # Get the number of reviews per game
    reviews_per_game = steam_reviews['app_name'].value_counts().reset_index()
    reviews_per_game.columns = ['app_name', 'number_of_reviews']

    app_ids = steam_reviews[['app_name', 'app_id']].drop_duplicates()

    reviews_per_game = reviews_per_game.merge(app_ids, on='app_name')

    for _, row in reviews_per_game.head(15).iterrows():
        print(f"{row['app_name']}, {row['app_id']}, {row['number_of_reviews']}")

    #Print the name and id of GTA V, Firewatch and Sonic Mania
    gta_v = steam_reviews[steam_reviews['app_name'].str.contains("Grand Theft Auto", case = False)].head(1)
    firewatch = steam_reviews[steam_reviews['app_name'].str.contains("Firewatch", case = False)].head(1)
    sonic_mania = steam_reviews[steam_reviews['app_name'].str.contains("Sonic Mania", case = False)].head(1)

    print(f"GTA V: {gta_v['app_name'].values[0]}, {gta_v['app_id'].values[0]}")
    print(f"Firewatch: {firewatch['app_name'].values[0]}, {firewatch['app_id'].values[0]}")
    print(f"Sonic Mania: {sonic_mania['app_name'].values[0]}, {sonic_mania['app_id'].values[0]}")


#Clustering the users to reduce the number of users
def main_5(n_clusters = 20000):
    # Load the data
    df = pd.read_csv('data/steam_reviews_all.csv')

    data = df[["recommended", "neg", "neu", "pos"]]

    data.loc[:, "recommended"] = data["recommended"].astype(bool)
    data.loc[:, "recommended"] = data["recommended"].astype(int)

    print(f"First 5 rows of the data:\n {data.head()}")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    hdb = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean', cluster_selection_method='eom')
    hdb.fit(scaled_data)

    df["cluster_id"] = hdb.labels_

    df['cluster_id'] = pd.factorize(df['cluster_id'])[0]

    print(f"First 5 rows of the data with the cluster IDs:\n {df.head()}")

    start_time = time.time()

    score = 0

    print("Number of clusters:", len(set(hdb.labels_)))
    print("Calculating silhouette score...")

    if len(set(hdb.labels_)) > 1:  # Ensure there's more than one cluster
        score = silhouette_score(scaled_data, hdb.labels_)
        print(f"Silhouette score: {score}")

    else:
        print("No valid clusters found.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for silhouette score calculation: {elapsed_time:.2f} seconds")

    if score < 0.2:
        print("Silhouette score is too low. Exiting.")
        exit()

    df.to_csv('data/steam_reviews_user_cluster_scan.csv', index = False)

    print("Clustering complete. User IDs and Cluster IDs saved as 'steam_reviews_user_cluster_scan.csv'.")

    # Transform the timestamps, so now they start form 0
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()

    print(f"First 5 rows of the data after the changes of the timestamp: {df.head()}")

    network_df = transform_to_network(df)
    print(f"First 5 rows of the network data:\n {network_df.head()}")

    network_df.rename(columns={'negative': 'comma_separated_list_of_features', 'neutral': '', 'positive': ''}, inplace=True)

    network_df.to_csv('data/steam_cluster_scan.csv', index=False)

    print("Data saved to 'data/steam_cluster_scan.csv'")


# Transform only the timestamps
def main_6():
    # Load the data
    df = pd.read_csv('data/steam_reviews_all.csv')

    # Transform the timestamps, so now they start form 0
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()

    print(f"First 5 rows of the data after the changes of the timestamp: {df.head()}")

    network_df = transform_to_network(df)
    print(f"First 5 rows of the network data: {network_df.head()}")

    network_df.rename(columns={'negative': 'comma_separated_list_of_features', 'neutral': '', 'positive': ''}, inplace=True)

    network_df.to_csv('data/steam_normalized_timestamps.csv', index=False)

    print("Data saved to 'data/steam_normalized_timestamps.csv'")


# Taking only the timestamps between January 1, 2017 and December 31, 2017
def main_7():
    # Load the data
    df = pd.read_csv('data/steam_2017.csv')

    # Filter out the timestamps between April 1, 2017 and September 30, 2017
    df = df[(df['timestamp'] >= 1491004800) & (df['timestamp'] <= 1506729599)]

    df.to_csv('data/steam_filtered_timestamps_2017.csv', index=False)


# Exploration of new data of games and reviews
def main_8():
    # Load the data
    fw = pd.read_csv('data/383870_Firewatch_Reviews.csv')
    sm = pd.read_csv('data/584400_Sonic_Mania_Reviews.csv')

    print(f"First 5 rows of the Firewatch data:\n {fw.head()}")

    print(f"First 5 rows of the Sonic Mania data:\n {sm.head()}")

    print(f"Columns in the Firewatch data: {fw.columns.tolist()}")
    print(f"Columns in the Sonic Mania data: {sm.columns.tolist()}")

    # Add the app_id and app_name columns
    sm['app_id'] = 584400
    sm['app_name'] = 'Sonic Mania'

    fw['app_id'] = 383870
    fw['app_name'] = 'Firewatch'

    # Filter out non-english reviews
    sm = sm[(sm["language"] == "english")]
    fw = fw[(fw["language"] == "english")]

    # Merge the two DataFrames
    all_reviews = pd.concat([sm, fw])
    # Sort the reviews by timestamp_created and timestamp_updated
    all_reviews = all_reviews.sort_values(by=['timestamp_created', 'timestamp_updated'], ascending=[True, True])

    print(f"First 5 rows of the merged data:\n {all_reviews.head()}")

    steam_reviews = pd.read_csv('data/steam_reviews.csv')

    # Filter out non-english reviews
    steam_reviews_english = steam_reviews[(steam_reviews["language"] == "english")]
    # Keep only the columns we need
    steam_reviews_english = steam_reviews_english[['app_id', 'app_name', 'review', 'timestamp_created', 'timestamp_updated', 'recommended', 'author.steamid', 'weighted_vote_score']]
    # Rename the column author.steamid to author_id
    steam_reviews_english.rename(columns={'author.steamid': 'author_id'}, inplace=True)

    # Merge the two DataFrames
    all_reviews = pd.concat([all_reviews, steam_reviews_english])
    # Sort the reviews by timestamp_created and timestamp_updated
    all_reviews = all_reviews.sort_values(by=['timestamp_created', 'timestamp_updated'], ascending=[True, True])

    print(f"First 5 rows of the merged data:\n {all_reviews.head()}")

    # Filter the reviews between January 1, 2017 and December 31, 2017
    all_reviews = all_reviews[(all_reviews['timestamp_created'] >= 1483228800) & (all_reviews['timestamp_created'] <= 1514764799)]

    print(f"Number of reviews in 2017: {all_reviews.shape[0]}")

    all_reviews.to_csv('data/steam_firewatch_sonic_reviews_2017.csv', index=False)

def main_9():
    df = pd.read_csv('data/steam_firewatch_sonic_reviews_2017.csv')

    # Merge the columns weighted_review_score and weighted_vote_score
    df['weighted_vote_score'] = df[['weighted_review_score', 'weighted_vote_score']].max(axis=1)
    df.drop(columns=['weighted_review_score'], inplace=True)

    df.fillna({'review': ''}, inplace=True)
    df.fillna({'weighted_vote_score': 0}, inplace=True)
    df.fillna({'language': 'english'}, inplace=True)

    print(f"No. of rows in the data: {df.shape[0]}")
    print(f"First 10 rows of the data:\n {df.head(10)}")

    # Get updated and not updated reviews
    steam_reviews_updated = get_updated_steam_reviews(df)
    steam_reviews_not_updated = get_not_updated_steam_reviews(df)

    # Create review DataFrame
    steam_reviews_updated = create_review_df(steam_reviews_updated)
    print(f"Number of updated reviews: {steam_reviews_updated.shape[0]}")
    steam_reviews_not_updated = create_review_df(steam_reviews_not_updated)
    print(f"Number of not updated reviews: {steam_reviews_not_updated.shape[0]}")

    new_df = merge_and_order_reviews(steam_reviews_updated, steam_reviews_not_updated)

    new_df.fillna({'review': ''}, inplace=True)
    new_df.fillna({'weighted_vote_score': 0}, inplace=True)
    new_df.fillna({'language': 'english'}, inplace=True)

    # Replace the author_id with smaller numbers starting from 0
    new_df['author_id'] = pd.factorize(new_df['author_id'])[0]

    # Replace the app_id with smaller numbers starting from 0
    new_df['app_id'] = pd.factorize(new_df['app_id'])[0]

    print(f"First 5 rows of the new data:\n {new_df.head()}")

    # Get sentiment scores using Roberta model
    sentiment_scores = get_sentiment_scores_roberta(new_df)
    print(f"First 5 sentiment scores: {sentiment_scores[:5]}")

    # Add sentiment scores to DataFrame
    steam_reviews_all_with_sentiment = add_sentiment_scores_to_df(new_df, sentiment_scores)

    # Save the final DataFrame to a CSV file
    steam_reviews_all_with_sentiment.to_csv('data/steam_reviews_roberta_2017_new.csv', index=False)

    print("Data saved to 'data/steam_reviews_roberta_2017_new.csv'")

# Plotting graphs to caracterize the data
def main_10():
    df = pd.read_csv('data/steam_reviews_all_2017_new.csv')

    # Only keep the reviews from 2017
    df = df[(df['timestamp'] >= 1483228800) & (df['timestamp'] <= 1514764799)]

    # Plot the number of the reviews of 7 games
    games = ['Grand Theft Auto V', 'Firewatch', 'Sonic Mania', 
             'PLAYERUNKNOWN\'S BATTLEGROUNDS', 'Doki Doki Literature Club', 
             'Tom Clancy\'s Rainbow Six Siege', 'Rocket League']
    
    seven_games = df[df['app_name'].isin(games)]

    print(f"Number of reviews of every game:\n {seven_games['app_name'].value_counts()}")

    one_week_unix = 604800

    for game in games:
        game_reviews = df[df['app_name'] == game]
        game_reviews['biweekly'] = (game_reviews['timestamp'] // (one_week_unix * 2)) * (one_week_unix * 2)
        biweekly_reviews = game_reviews.groupby('biweekly').size()
        biweekly_recommended = game_reviews[game_reviews['recommended'] == True].groupby('biweekly').size()
        biweekly_percentage_recommended = (biweekly_recommended / biweekly_reviews)
        biweekly_mean_neg = game_reviews.groupby('biweekly')['neg'].mean()

        fig, ax1 = plt.subplots(figsize=(18, 6))

        ax1.set_xlabel('Biweekly Period')
        ax1.set_ylabel('Mean Sentiment Scores')
        ax1.plot(biweekly_mean_neg.index, biweekly_mean_neg, marker='o', linestyle='-', label='Mean Negative Score', color='tab:red')
        ax1.tick_params(axis='y')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        if game == 'Grand Theft Auto V':
            ax1.axvspan(1496268000, 1501538399, color='red', alpha=0.3)

        if game == 'Firewatch':
            ax1.axvspan(1504224000, 1506815999, color='red', alpha=0.3)
        
        if game == 'Sonic Mania':
            ax1.axvspan(1501545600, 1504223999, color='red', alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Percentage of Recommended Reviews')
        ax2.plot(biweekly_percentage_recommended.index, biweekly_percentage_recommended, marker='o', linestyle='-', label='Percentage of Recommended Reviews', color='tab:blue')
        ax2.tick_params(axis='y')
        ax2.legend(loc='upper right')

        plt.title(f'Biweekly Mean Sentiment Scores and Percentage of Recommended Reviews for {game} in 2017')
        fig.tight_layout()
        plt.savefig(f'biweekly_mean_sentiment_scores_and_percentage_recommended_{game.replace(" ", "_").replace("\'", "")}_2017.png')

def main_11():
    df = pd.read_csv('data/steam_2017_new.csv')

    # Print the size of the dataset
    print(f"Number of rows in the dataset: {df.shape[0]}")

    # Get the number of unique users
    unique_users = df['user_id'].nunique()
    print(f"Number of unique users: {unique_users}")

    # Get the number of unique games
    unique_games = df['item_id'].nunique()
    print(f"Number of unique games: {unique_games}")

    # Get the number of user with state_label 1
    state_label_1 = df[df['state_label'] == 1]
    unique_state_label_1 = state_label_1['user_id'].nunique()
    print(f"Number of unique users with state label 1: {unique_state_label_1}")

    # Get the number of user with state_label 0 excluding if the user has state_label 1
    state_label_0 = df[(df['state_label'] == 0) & (~df['user_id'].isin(state_label_1['user_id']))]
    unique_state_label_0 = state_label_0['user_id'].nunique()
    print(f"Number of unique users with state label 0: {unique_state_label_0}")

    # Get the number of games with state_label 1
    unique_state_label_1_games = state_label_1['item_id'].nunique()
    print(f"Number of unique games with state label 1: {unique_state_label_1_games}")

def main_12(file_name = '/home/bigdama/projects/tgn/results/tgn-attn-steam'):

    files_tgn = [file_name + '.pkl', file_name + '_1.pkl', file_name + '_2.pkl', file_name +
                  '_3.pkl', file_name + '_4.pkl', file_name + '_5.pkl', file_name + '_6.pkl', 
                  file_name + '_7.pkl', file_name + '_8.pkl', file_name + '_9.pkl']
    
    results = []
    for file in files_tgn:
        with open(file, 'rb') as f:
            results.append(pickle.load(f))

    keys = results[0].keys()
    results

    # Initialize a dictionary to store the means
    means = {key: [] for key in keys}

    # Loop through each result and calculate the mean for each key
    for result in results:
        for key in keys:
            means[key].append(np.mean(result[key]))

    # Calculate the overall mean for each key
    overall_means = {key: np.mean(means[key]) for key in keys}

    # Calculate the overall mean plus/minus variation for each key
    overall_means_variation = {key: (np.mean(means[key]), np.std(means[key])) for key in keys}

    # Transform the val_aps and test_ap into percentage format
    overall_means_variation['val_aps'] = (overall_means_variation['val_aps'][0] * 100, overall_means_variation['val_aps'][1] * 100)
    overall_means_variation['test_ap'] = (overall_means_variation['test_ap'][0] * 100, overall_means_variation['test_ap'][1] * 100)

    print(overall_means_variation)

def main_13(path = '/home/bigdama/projects/tgn/data/steam_2017_new.csv'):
    df = pd.read_csv(path)

    # Add two and a half months to the timestamp of the reviews with state_label 1 and item_id 14 (GTA V)
    df.loc[(df['state_label'] == 1) & (df['item_id'] == 14), 'timestamp'] += 5443200

    # Get how many reviews have been modified
    modified_reviews = df[(df['state_label'] == 1) & (df['item_id'] == 14)]
    print(f"Number of modified reviews: {modified_reviews.shape[0]}")

    # Get the range of the timestamps of the modified reviews
    print(f"Range of the timestamps of the modified reviews: {modified_reviews['timestamp'].min()} - {modified_reviews['timestamp'].max()}")

    # Move two and a half months the same amount of modified reviews from the reviews with state_label 0 and item_id 14 (GTA V) in the same range of timestamps
    df.loc[(df['state_label'] == 0) & (df['item_id'] == 14) & (df['timestamp'] >= modified_reviews['timestamp'].min()) & 
           (df['timestamp'] <= modified_reviews['timestamp'].max()), 'timestamp'] -= 5443200


    # Substract one and a half months to the timestamp of the reviews with state_label 1 and item_id 1 (Firewatch)
    df.loc[(df['state_label'] == 1) & (df['item_id'] == 1), 'timestamp'] -= 3888000

    # Get how many reviews have been modified
    modified_reviews = df[(df['state_label'] == 1) & (df['item_id'] == 1)]
    print(f"Number of modified reviews: {modified_reviews.shape[0]}")

    # Get the range of the timestamps of the modified reviews
    print(f"Range of the timestamps of the modified reviews: {modified_reviews['timestamp'].min()} - {modified_reviews['timestamp'].max()}")

    # Move two and a half months the same amount of modified reviews from the reviews with state_label 0 and item_id 1 (Firewatch) in the same range of timestamps
    df.loc[(df['state_label'] == 0) & (df['item_id'] == 1) & (df['timestamp'] >= modified_reviews['timestamp'].min()) &
              (df['timestamp'] <= modified_reviews['timestamp'].max()), 'timestamp'] += 3888000
    
    # Sort the reviews by timestamp
    df = df.sort_values(by=['timestamp'], ascending=[True])

    df.to_csv('data/steam_2017_new_modified.csv', index=False)
    print("Data saved to 'data/steam_2017_new_modified.csv'")
    df.to_csv('/home/bigdama/projects/tgn/data/steam_2017_new_modified.csv', index=False)
    print("Data saved to '/home/bigdama/projects/tgn/data/steam_2017_new_modified.csv'")
    df.to_csv('/home/bigdama/projects/bidyn/data/steam_2017_new_modified.csv', index=False)
    print("Data saved to '/home/bigdama/projects/bidyn/data/steam_2017_new_modified.csv'")

def main_14():

    df = pd.read_csv('data/steam_2017_new_modified.csv')

    print(f"First 30 rows of the data before the swap:\n {df[['user_id', 'item_id', 'timestamp', 'state_label']].head(30)}")

    # swap the columns user_id and item_id
    df.rename(columns={'user_id': 'item_id', 'item_id': 'user_id'}, inplace=True)

    df = df[['user_id', 'item_id'] + [col for col in df.columns if col not in ['user_id', 'item_id']]]

    print(f"First 30 rows of the data after the swap:\n {df[['user_id', 'item_id', 'timestamp', 'state_label']].head(30)}")

    df.to_csv('data/steam_2017_new_swapped.csv', index=False)

def main_15():
    steam_reviews = pd.read_csv('data/steam_reviews_all_2017_new.csv')

    # Get the users were review_bombing is 1
    review_bombing_users = steam_reviews[steam_reviews['review_bombing'] == 1]

    print(f"Number of review bombing users: {review_bombing_users.shape[0]}")

    # Get the users were review_bombing is 0
    non_review_bombing_users = steam_reviews[steam_reviews['review_bombing'] == 0]

    # Get the mean for the negative, neutral and positive sentiment scores for the review
    review_bombing_mean = review_bombing_users[['neg', 'neu', 'pos']].mean()
    non_review_bombing_mean = non_review_bombing_users[['neg', 'neu', 'pos']].mean()

    # Save an histogram of the scores
    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.35
    index = np.arange(3)

    ax.bar(index, review_bombing_mean, bar_width, label='Review Bombing', color='red', alpha=0.7)
    ax.bar(index + bar_width, non_review_bombing_mean, bar_width, label='Non Review Bombing', color='blue', alpha=0.7)

    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Mean Sentiment Score')
    ax.set_title('Mean Sentiment Scores for Review Bombing user v/s Non Review Bombing user')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    ax.legend()

    # Save the plot as a PNG file
    plt.savefig('mean_sentiment_scores_review_bombing_v_non_review_bombing.png')

def main_16():
    path = "/home/bigdama/projects/bidyn/out/embs.pt"

    data = torch.load(path)

    print_data_info(Data(x=data[0]))

    print_data_info(Data(x=data[1]))

def main_17():
    path = "/home/bigdama/projects/bidyn/out/pred.pt"

    with open(path, "rb") as f:
        preds = pickle.load(f)
        u_embs_abusive = preds['u_embs_abusive']
        mean_non_abusive = preds['mean_non_abusive']
        cos_sim_abusive = preds['cos_sim_abusive']
        cos_sim_non_abusive = preds['cos_sim_non_abusive']

    avg_cos_sim_abusive = np.mean(cos_sim_abusive)
    avg_cos_sim_non_abusive = np.mean(cos_sim_non_abusive)
    print(f"Average cosine similarity within abusive users: {avg_cos_sim_abusive}")
    print(f"Average cosine similarity within non-abusive users: {avg_cos_sim_non_abusive}")

    std_cos_sim_abusive = np.std(cos_sim_abusive)
    std_cos_sim_non_abusive = np.std(cos_sim_non_abusive)
    print(f"Standard deviation of cosine similarity within abusive users: {std_cos_sim_abusive}")
    print(f"Standard deviation of cosine similarity within non-abusive users: {std_cos_sim_non_abusive}")

    kmeans = KMeans(n_clusters=5, random_state=42)
    abusive_clusters = kmeans.fit_predict(u_embs_abusive)

    abusive_user_clusters = pd.DataFrame(u_embs_abusive, columns=[f'feature_{i}' for i in range(u_embs_abusive.shape[1])])
    abusive_user_clusters['cluster'] = abusive_clusters

    print(f"Cluster labels for abusive users:\n {abusive_user_clusters['cluster'].value_counts()}")

    cluster_cos_similarities = {}
    for cluster in abusive_user_clusters['cluster'].unique():
        cluster_users = abusive_user_clusters[abusive_user_clusters['cluster'] == cluster].drop(columns=['cluster'])
        cos_sim_matrix = sklearn_cosine_similarity(cluster_users)
        avg_cos_sim = np.mean(cos_sim_matrix)
        cluster_cos_similarities[cluster] = avg_cos_sim

    print(f"Cosine similarity between users of the same cluster:\n {cluster_cos_similarities}")

    cluster_centroids = kmeans.cluster_centers_
    euclidean_distances = np.linalg.norm(cluster_centroids - mean_non_abusive, axis=1)

    for i, distance in enumerate(euclidean_distances):
        print(f"Euclidean distance between cluster {i} centroid and the mean embedding of non-abusive users: {distance}")

def main_18():
    path = "/home/bigdama/projects/bidyn/out/preds.pt"

    with open(path, "rb") as f:
        preds = pickle.load(f)
        train_logp = preds['train_logp']
        train_labels = preds['train_labels']
        val_logp = preds['val_logp']
        val_labels = preds['val_labels']
        test_logp = preds['test_logp']
        test_labels = preds['test_labels']
        edge_features = preds['edge_features']
        train_feats = preds['train_feats']
        val_feats = preds['val_feats']
        test_feats = preds['test_feats']
        u_embs = preds['u_embs']
        u_embs_np = preds['u_embs_np']
        u_embs_abusive = preds['u_embs_abusive']
        u_embs_non_abusive = preds['u_embs_non_abusive']
        mean_abusive = preds['mean_abusive']
        std_abusive = preds['std_abusive']
        mean_non_abusive = preds['mean_non_abusive']
        std_non_abusive = preds['std_non_abusive']
        cos_sim_abusive = preds['cos_sim_abusive']
        mean_sim_abusive = preds['mean_sim_abusive']
        cos_sim_non_abusive = preds['cos_sim_non_abusive']
        mean_sim_non_abusive = preds['mean_sim_non_abusive']
        bad_edges = preds['bad_edges']
        bad_edges_feats = preds['bad_edges_feats']

    print(f"u_embs_abusive length: {len(u_embs_abusive)}")

    train_preds = torch.exp(torch.tensor(train_logp))
    val_preds = torch.exp(torch.tensor(val_logp))
    test_preds = torch.exp(torch.tensor(test_logp))

    train_preds_vals = extract_highest_probability_val(train_preds)
    val_preds_vals = extract_highest_probability_val(val_preds)
    test_preds_vals = extract_highest_probability_val(test_preds)

    train_labels_counts = pd.Series(train_labels).value_counts()
    val_labels_counts = pd.Series(val_labels).value_counts()
    test_labels_counts = pd.Series(test_labels).value_counts()

    print(f"Train labels value counts:\n{train_labels_counts}")
    print(f"Validation labels value counts:\n{val_labels_counts}")
    print(f"Test labels value counts:\n{test_labels_counts}")

    i = 0
    train_preds_with_features = []
    for feat in tqdm(train_feats, desc="Train", total=len(train_feats)):
        if len(feat) > 0:
            if any(bad_feat in bad_edges_feats for bad_feat in feat):
                train_preds_with_features.append((1, feat, train_preds_vals[i], train_labels[i]))
            else:
                train_preds_with_features.append((0, feat, train_preds_vals[i], train_labels[i]))
            i += 1
    
    i = 0
    val_preds_with_features = []
    for feat in tqdm(val_feats, desc="Validation", total=len(val_feats)):
        if len(feat) > 0:
            if any(bad_feat in bad_edges_feats for bad_feat in feat):
                val_preds_with_features.append((1, feat, val_preds_vals[i], val_labels[i]))
            else:
                val_preds_with_features.append((0, feat, val_preds_vals[i], val_labels[i]))
            i += 1
    i = 0
    test_preds_with_features = []
    for feat in tqdm(test_feats, desc="Test", total=len(test_feats)):
        if len(feat) > 0:
            if any(bad_feat in bad_edges_feats for bad_feat in feat):
                test_preds_with_features.append((1, feat, test_preds_vals[i], test_labels[i]))
            else:
                test_preds_with_features.append((0, feat, test_preds_vals[i], test_labels[i]))
            i += 1

    all_preds = train_preds_with_features + val_preds_with_features + test_preds_with_features

    correct_predictions = 0
    for i, (pred, _, _, _) in enumerate(train_preds_with_features):
        if pred == train_preds_vals[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(train_preds_vals)
    print(f"Accuracy of train_preds_with_features compared to train_preds_vals: {accuracy:.2f}")

    correct = 0
    incorrect_one = 0
    for i, (pred, _, _, _) in enumerate(train_preds_with_features):
        if pred == train_labels[i]:
            correct += 1
        elif pred == 1:
            incorrect_one += 1
    
    accuracy = correct / len(train_labels)
    print(f"Accuracy of train_preds_with_features compared to train_labels: {accuracy:.2f}")
    print(f"Number of incorrect predictions 1: {incorrect_one}")

    correct_predictions = 0
    for i, (pred, _, _, _) in enumerate(val_preds_with_features):
        if pred == val_preds_vals[i]:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(val_preds_vals)
    print(f"Accuracy of val_preds_with_features compared to val_preds_vals: {accuracy:.2f}")

    correct = 0
    incorrect_one = 0
    for i, (pred, _, _, _) in enumerate(val_preds_with_features):
        if pred == val_labels[i]:
            correct += 1
        elif pred == 1:
            incorrect_one += 1
    
    accuracy = correct / len(val_labels)
    print(f"Accuracy of val_preds_with_features compared to val_labels: {accuracy:.2f}")
    print(f"Number of incorrect predictions 1: {incorrect_one}")

    correct_predictions = 0
    for i, (pred, _, _, _) in enumerate(test_preds_with_features):
        if pred == test_preds_vals[i]:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(test_preds_vals)
    print(f"Accuracy of test_preds_with_features compared to test_preds_vals: {accuracy:.2f}")

    correct = 0
    incorrect_one = 0
    for i, (pred, _, _, _) in enumerate(test_preds_with_features):
        if pred == test_labels[i]:
            correct += 1
        elif pred == 1:
            incorrect_one += 1
    
    accuracy = correct / len(test_labels)
    print(f"Accuracy of test_preds_with_features compared to test_labels: {accuracy:.2f}")
    print(f"Number of incorrect predictions 1: {incorrect_one}")

    preds_zero = [pred for pred, features, _, _ in all_preds if pred == 0]
    preds_one = [pred for pred, features, _, _ in all_preds if pred == 1]

    print(f"Number of predictions 0: {len(preds_zero)}")
    print(f"Number of predictions 1: {len(preds_one)}")

    zero_features = [features for pred, features, _, _ in all_preds if pred == 0]
    one_features = [features for pred, features, _, _ in all_preds if pred == 1]

    mean_zero_features = [np.mean(features, axis=0) for features in zero_features]
    mean_one_features = [np.mean(features, axis=0) for features in one_features]

    mean_zero_features = np.mean(mean_zero_features, axis=0)
    mean_one_features = np.mean(mean_one_features, axis=0)

    print(f"Mean features for predictions 0: {mean_zero_features}")
    print(f"Mean features for predictions 1: {mean_one_features}")

    preds_zero = [pred for _, _, pred, _ in all_preds if pred == 0]
    preds_one = [pred for _, _, pred, _ in all_preds if pred == 1]

    print(f"Numbre of predictions 0: {len(preds_zero)}")
    print(f"Number of predictions 1: {len(preds_one)}")

    zero_features = [features for _, features, pred, _ in all_preds if pred == 0]
    one_features = [features for _, features, pred, _ in all_preds if pred == 1]

    mean_zero_features = [np.mean(features, axis=0) for features in zero_features]
    mean_one_features = [np.mean(features, axis=0) for features in one_features]

    mean_zero_features = np.mean(mean_zero_features, axis=0)
    mean_one_features = np.mean(mean_one_features, axis=0)

    print(f"Mean features for predictions 0: {mean_zero_features}")
    print(f"Mean features for predictions 1: {mean_one_features}")

    preds_zero = [pred for _, _, _, pred in all_preds if pred == 0]
    preds_one = [pred for _, _, _, pred in all_preds if pred == 1]

    print(f"Number of predictions 0: {len(preds_zero)}")
    print(f"Number of predictions 1: {len(preds_one)}")

    zero_features = [features for _, features, _, pred in all_preds if pred == 0]
    one_features = [features for _, features, _, pred in all_preds if pred == 1]

    mean_zero_features = [np.mean(features, axis=0) for features in zero_features]
    mean_one_features = [np.mean(features, axis=0) for features in one_features]

    mean_zero_features = np.mean(mean_zero_features, axis=0)
    mean_one_features = np.mean(mean_one_features, axis=0)

    print(f"Mean features for predictions 0: {mean_zero_features}")
    print(f"Mean features for predictions 1: {mean_one_features}")

def main_19():
    path = "/home/bigdama/projects/bidyn/out/preds_1.pt"

    with open(path, "rb") as f:
        preds = pickle.load(f)
        u_labels = preds['u_labels']
        u_to_idx = preds['u_to_idx']
        train_us = preds['train_us']
        u_train_mask = preds['u_train_mask']
        u_val_mask = preds['u_val_mask']
        u_test_mask = preds['u_test_mask']
        d_labels = preds['d_labels']
        train_logp = preds['train_logp']
        train_labels = preds['train_labels']
        val_logp = preds['val_logp']
        val_labels = preds['val_labels']
        test_logp = preds['test_logp']
        test_labels = preds['test_labels']
        edge_features = preds['edge_features']
        train_feats = preds['train_feats']
        val_feats = preds['val_feats']
        test_feats = preds['test_feats']
        u_embs = preds['u_embs']
        u_embs_np = preds['u_embs_np']
        u_embs_abusive = preds['u_embs_abusive']
        u_embs_non_abusive = preds['u_embs_non_abusive']
        mean_abusive = preds['mean_abusive']
        std_abusive = preds['std_abusive']
        mean_non_abusive = preds['mean_non_abusive']
        std_non_abusive = preds['std_non_abusive']
        cos_sim_abusive = preds['cos_sim_abusive']
        mean_sim_abusive = preds['mean_sim_abusive']
        cos_sim_non_abusive = preds['cos_sim_non_abusive']
        mean_sim_non_abusive = preds['mean_sim_non_abusive']
        bad_edges = preds['bad_edges']
        bad_edges_feats = preds['bad_edges_feats']
    
    train_preds = torch.exp(torch.tensor(train_logp))
    val_preds = torch.exp(torch.tensor(val_logp))
    test_preds = torch.exp(torch.tensor(test_logp))

    train_preds_vals = extract_highest_probability_val(train_preds)
    val_preds_vals = extract_highest_probability_val(val_preds)
    test_preds_vals = extract_highest_probability_val(test_preds)

    print("u_to_idx length:", len(u_to_idx))
    print("u_labels length:", len(u_labels))
    print("d_labels length:", len(d_labels))
    print("train_feats length:", len(train_feats))

    train_mask = [None for _ in range(len(u_train_mask))]
    for u, i in u_to_idx.items():
        train_mask[u] = u_train_mask[i]
    
    val_mask = [None for _ in range(len(u_val_mask))]
    for u, i in u_to_idx.items():
        val_mask[u] = u_val_mask[i]

    test_mask = [None for _ in range(len(u_test_mask))]
    for u, i in u_to_idx.items():
        test_mask[u] = u_test_mask[i]

    i = 0
    j = 0
    train_preds_labels = [None for _ in range(len(train_mask))]
    for bool_val in train_mask:
        if bool_val:
            train_preds_labels[i] = train_preds_vals[j]
            j += 1
        i += 1
    
    i = 0
    j = 0
    val_preds_labels = [None for _ in range(len(val_mask))]
    for bool_val in val_mask:
        if bool_val:
            val_preds_labels[i] = val_preds_vals[j]
            j += 1
        i += 1
    
    i = 0
    j = 0
    test_preds_labels = [None for _ in range(len(test_mask))]
    for bool_val in test_mask:
        if bool_val:
            test_preds_labels[i] = test_preds_vals[j]
            j += 1
        i += 1
        
    new_u_labels = [[] for _ in range(len(train_feats))]
    new_train_labels = [[] for _ in range(len(train_feats))]
    for u, i in u_to_idx.items():
        if len(train_feats[i]) > 0:
            new_train_labels[i] = train_preds_labels[u]
            new_u_labels[i] = d_labels[u]
    
    new_val_labels = [[] for _ in range(len(val_feats))]
    for u, i in u_to_idx.items():
        if len(val_feats[i]) > 0:
            new_val_labels[i] = val_preds_labels[u]
            new_u_labels[i] = d_labels[u]

    
    new_test_labels = [[] for _ in range(len(test_feats))]
    for u, i in u_to_idx.items():
        if len(test_feats[i]) > 0:
            new_test_labels[i] = test_preds_labels[u]
            new_u_labels[i] = d_labels[u]

    u_labels = u_labels.tolist()

    # Compare all elements of u_labels and new_u_labels
    comparison_result = all(ul == nul for ul, nul in zip(u_labels, new_u_labels))

    print(f"Are all elements of u_labels and new_u_labels equal? {comparison_result}")

    new_train_labels = [label for label in new_train_labels if isinstance(label, np.int64)]
    new_val_labels = [label for label in new_val_labels if isinstance(label, np.int64)]
    new_test_labels = [label for label in new_test_labels if isinstance(label, np.int64)]

    i = 0    
    train_preds_with_features = []
    for feat in tqdm(train_feats, desc="Train", total=len(train_feats)):
        if len(feat) > 0:
            train_preds_with_features.append((feat, new_train_labels[i]))
            i += 1
    
    i = 0
    val_preds_with_features = []
    for feat in tqdm(val_feats, desc="Validation", total=len(val_feats)):
        if len(feat) > 0:
            val_preds_with_features.append((feat, new_val_labels[i]))
            i += 1

    i = 0
    test_preds_with_features = []
    for feat in tqdm(test_feats, desc="Test", total=len(test_feats)):
        if len(feat) > 0:
            test_preds_with_features.append((feat, new_test_labels[i]))
            i += 1

    
    all_preds = train_preds_with_features + val_preds_with_features + test_preds_with_features

    preds_zero = [pred for _, pred in all_preds if pred == 0]
    preds_one = [pred for _, pred in all_preds if pred == 1]

    print(f"Number of predictions 0: {len(preds_zero)}")
    print(f"Number of predictions 1: {len(preds_one)}")

    zero_features = [features for features, pred in all_preds if pred == 0]
    one_features = [features for features, pred in all_preds if pred == 1]

    mean_zero_features = [np.mean(features, axis=0) for features in zero_features]
    mean_one_features = [np.mean(features, axis=0) for features in one_features]

    mean_zero_features = np.mean(mean_zero_features, axis=0)
    mean_one_features = np.mean(mean_one_features, axis=0)

    print(f"Mean features for predictions 0: {mean_zero_features}")
    print(f"Mean features for predictions 1: {mean_one_features}")
    
if __name__ == "__main__":
    main_19()

