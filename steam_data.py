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
import zipfile
#Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples
import time



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

    # Get sentiment scores using Roberta model
    sentiment_scores = get_sentiment_scores_roberta(steam_reviews_all)
    print(f"First 5 sentiment scores: {sentiment_scores[:5]}")

    # Add sentiment scores to DataFrame
    steam_reviews_all_with_sentiment = add_sentiment_scores_to_df(steam_reviews_all, sentiment_scores)

    # Save the final DataFrame to a CSV file
    steam_reviews_all_with_sentiment.to_csv('data/steam_reviews_roberta.csv', index=False)


# Marking with a 1 the reviews that mention Take-Two or OpenIV in a new column called 'review_bombing'
def main_2():
    steam_reviews = pd.read_csv('data/steam_reviews_roberta.csv')

    #Possible review Bombing for GTA V between 2017-06-01 and 2017-07-31
    one_game_only_english = steam_reviews[(steam_reviews["app_name"].str.contains("Grand Theft Auto", case = False))
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
    
    print(f"Number of reviews that mention Take-Two, OpenIV, modding, mod or Rockstar: {one_game_only_english.shape[0]}")

    # Create a DataFrame excluding the values in one_game_only_english
    steam_reviews_excluding_bombing = steam_reviews[~steam_reviews.index.isin(one_game_only_english.index)]

    steam_reviews_excluding_bombing = steam_reviews_excluding_bombing.assign(review_bombing=0)

    one_game_only_english = one_game_only_english.assign(review_bombing=1)

    # Merge the two DataFrames
    steam_reviews_all = merge_and_order_reviews(steam_reviews_excluding_bombing, one_game_only_english)

    steam_reviews_all.to_csv('data/steam_reviews_all.csv', index=False)

    network_df = transform_to_network(steam_reviews_all)
    print(f"First 5 rows of the network data:\n {network_df.head()}")

    network_df.rename(columns={'negative': 'comma_separated_list_of_features', 'neutral': '', 'positive': ''}, inplace=True)

    network_df.to_csv('data/steam.csv', index=False)

    print("Data saved to 'data/steam.csv'")

    # Create a Zip file
    with zipfile.ZipFile('data/steam_data_complete.zip', 'w') as zipf:
        zipf.write('data/steam_reviews_roberta.csv', arcname='steam_reviews_roberta.csv')
        zipf.write('data/steam_reviews_all.csv', arcname='steam_reviews_all.csv')
        zipf.write('data/steam.csv', arcname='steam.csv')


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
    steam_reviews = pd.read_csv('data/steam_reviews.csv')

    # Filter out non-english reviews
    steam_reviews_english = steam_reviews[(steam_reviews["language"] == "english")]

    # Print the size of the dataset
    print(f"Number of rows in the dataset: {steam_reviews_english.shape[0]}")

    # Get the number of unique users
    unique_users = steam_reviews_english['author.steamid'].nunique()
    print(f"Number of unique users: {unique_users}")

    # Get the number of unique games
    unique_games = steam_reviews_english['app_id'].nunique()
    print(f"Number of unique games: {unique_games}")

    # Get the number of unique reviews
    unique_reviews = steam_reviews_english['review_id'].nunique()
    print(f"Number of unique reviews: {unique_reviews}")


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

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100000, verbose=True)
    kmeans.fit(scaled_data)

    df["cluster_id"] = kmeans.labels_

    df['cluster_id'] = pd.factorize(df['cluster_id'])[0]

    print(f"First 5 rows of the data with the cluster IDs:\n {df.head()}")

    start_time = time.time()

    silhouette_avg = silhouette_samples(scaled_data, kmeans.labels_)
    silhouette_avg_mean = np.mean(silhouette_avg)
    print("The mean silhouette_score is :", silhouette_avg_mean)
    print("The average silhouette_score is :", silhouette_avg)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for silhouette score calculation: {elapsed_time:.2f} seconds")

    if silhouette_avg < 0.2:
        print("The silhouette score is low. Try increasing the number of clusters.")
        exit()

    df.to_csv('data/steam_reviews_user_cluster.csv', index = False)

    print("Clustering complete. User IDs and Cluster IDs saved as 'steam_reviews_user_cluster.csv'.")

    # Transform the timestamps, so now they start form 0
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()

    print(f"First 5 rows of the data after the changes of the timestamp: {df.head()}")

    network_df = transform_to_network(df)
    print(f"First 5 rows of the network data: {network_df.head()}")

    network_df.rename(columns={'negative': 'comma_separated_list_of_features', 'neutral': '', 'positive': ''}, inplace=True)

    network_df.to_csv('data/steam_cluster.csv', index=False)

    print("Data saved to 'data/steam_cluster.csv'")


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

if __name__ == "__main__":
    main_5()

