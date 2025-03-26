# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical operations
import matplotlib.pyplot as plt  # For creating visualizations
from sklearn.preprocessing import MinMaxScaler  # For normalizing data

def load_data():
    """
    Load the MovieLens dataset from local CSV files.

    This function attempts to read the movies and ratings data from local CSV files.
    If the files are not found, it returns None values and displays an error message.

    Returns:
        movies_df (DataFrame): Contains movie information (movieId, title, genres)
        ratings_df (DataFrame): Contains user ratings (userId, movieId, rating, timestamp)
    """
    try:
        # Attempt to load data from local CSV files
        movies_df = pd.read_csv('movies.csv')
        ratings_df = pd.read_csv('ratings.csv')
        print("Data loaded from local files.")
    except FileNotFoundError:
        # Display error message if files are not found
        print("Local CSV files not found. Please ensure 'movies.csv' and 'ratings.csv' are in the same directory as this script.")
        return None, None

    return movies_df, ratings_df

def calculate_popularity(movies_df, ratings_df):
    """
    Calculate popularity scores for movies based on ratings data.

    This function computes a popularity score for each movie using:
    1. Average rating (weighted 40%)
    2. Number of ratings (weighted 60%)

    The weights favor movies with more ratings over those with higher but fewer ratings,
    which helps avoid recommending niche movies with only a few high ratings.

    Args:
        movies_df (DataFrame): Contains movie information
        ratings_df (DataFrame): Contains user ratings

    Returns:
        DataFrame: Movies with calculated popularity scores, sorted by popularity
    """
    # Calculate average rating for each movie
    avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating']

    # Calculate number of ratings for each movie
    rating_counts = ratings_df.groupby('movieId')['rating'].count().reset_index()
    rating_counts.columns = ['movieId', 'rating_count']

    # Merge average ratings and rating counts
    popularity_df = pd.merge(avg_ratings, rating_counts, on='movieId')

    # Normalize the values to a 0-1 scale using MinMaxScaler
    scaler = MinMaxScaler()
    popularity_df[['avg_rating_scaled', 'rating_count_scaled']] = scaler.fit_transform(
        popularity_df[['avg_rating', 'rating_count']])

    # Calculate weighted popularity score (40% rating, 60% number of ratings)
    popularity_df['popularity_score'] = (
        0.4 * popularity_df['avg_rating_scaled'] +
        0.6 * popularity_df['rating_count_scaled']
    )

    # Sort by popularity score in descending order
    popularity_df = popularity_df.sort_values('popularity_score', ascending=False)

    # Merge with movie information to get titles and genres
    result = pd.merge(popularity_df, movies_df, on='movieId')

    return result

def recommend_popular_movies(popularity_df, n=10, genre=None):
    """
    Recommend the top n popular movies, optionally filtered by genre.

    This function returns the most popular movies overall or within a specific genre.
    When a genre is specified, it filters the movies and returns the most popular ones
    in that genre.

    Args:
        popularity_df (DataFrame): DataFrame with movie popularity information
        n (int): Number of movies to recommend (default: 10)
        genre (str, optional): Genre to filter by (default: None)

    Returns:
        DataFrame: Top n recommended movies sorted by popularity score
    """
    if genre:
        # Filter movies by the specified genre
        filtered_df = popularity_df[popularity_df['genres'].str.contains(genre, case=False, na=False)]

        # Check if any movies were found for the genre
        if filtered_df.empty:
            print(f"No movies found for genre: {genre}")
            return None

        # Sort by popularity score to ensure we get the most popular movies in this genre
        filtered_df = filtered_df.sort_values('popularity_score', ascending=False)
        return filtered_df.head(n)
    else:
        # Return the top n most popular movies overall
        return popularity_df.head(n)

def visualize_popularity(popularity_df, top_n=20, genre=None):
    """
    Create a horizontal bar chart of the top N popular movies.

    This function visualizes the most popular movies overall or within a specific genre.
    It saves the visualization as a PNG file and displays a message with the filename.

    Args:
        popularity_df (DataFrame): DataFrame with movie popularity information
        top_n (int): Number of top movies to visualize (default: 20)
        genre (str, optional): Genre to filter by (default: None)
    """
    # Create a new figure with specified size
    plt.figure(figsize=(12, 8))

    if genre:
        # Filter by genre if specified
        filtered_df = popularity_df[popularity_df['genres'].str.contains(genre, case=False, na=False)]

        # Check if any movies were found for the genre
        if filtered_df.empty:
            print(f"No movies found for genre: {genre}")
            return

        # Sort by popularity score and get top movies
        filtered_df = filtered_df.sort_values('popularity_score', ascending=False)
        top_movies = filtered_df.head(top_n)

        # Set title and filename for genre-specific visualization
        title = f'Top {top_n} Popular {genre} Movies'
        filename = f'top_popular_{genre.lower()}_movies.png'
    else:
        # Get top movies overall
        top_movies = popularity_df.head(top_n)

        # Set title and filename for overall popularity visualization
        title = f'Top {top_n} Popular Movies'
        filename = 'top_popular_movies.png'

    # Create horizontal bar chart
    plt.barh(top_movies['title'], top_movies['popularity_score'])
    plt.xlabel('Popularity Score')
    plt.ylabel('Movie Title')
    plt.title(title)
    plt.tight_layout()  # Adjust layout to make room for labels

    # Save the figure and close it
    plt.savefig(filename)
    plt.close()

    print(f"Visualization saved as '{filename}'")

def get_available_genres(movies_df):
    """
    Extract all unique genres from the movies dataframe.

    This function parses the genres column, which contains pipe-separated genre lists,
    and returns a sorted list of all unique genres in the dataset.

    Args:
        movies_df (DataFrame): DataFrame containing movie information

    Returns:
        list: Sorted list of unique genres
    """
    # Initialize empty list to store all genres
    all_genres = []

    # Iterate through each movie's genres
    for genres in movies_df['genres'].str.split('|'):
        # Check if genres is a list (not NaN)
        if isinstance(genres, list):
            # Add all genres from this movie to the list
            all_genres.extend(genres)

    # Return sorted list of unique genres
    return sorted(list(set(all_genres)))

def get_user_input(available_genres):
    """
    Get user input for recommendation type and parameters.

    This function presents an interactive menu to the user, allowing them to choose
    between popularity-based recommendations or genre-based recommendations.
    It validates all user inputs and returns a dictionary with the user's preferences.

    Args:
        available_genres (list): List of available genres to choose from

    Returns:
        dict: Dictionary with user preferences including:
            - type: "popularity" or "genre"
            - n: number of recommendations
            - genre: selected genre (only if type is "genre")
    """
    # Display menu header
    print("\n=== Movie Recommendation System ===")
    print("1. Get recommendations by popularity")
    print("2. Get recommendations by genre")

    # Get and validate user choice (1 or 2)
    while True:
        try:
            choice = int(input("\nEnter your choice (1 or 2): "))
            if choice in [1, 2]:
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Handle popularity-based recommendation choice
    if choice == 1:
        # Get and validate number of recommendations
        while True:
            try:
                n = int(input("How many recommendations would you like? "))
                if n > 0:
                    return {"type": "popularity", "n": n}
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    # Handle genre-based recommendation choice
    else:
        # Display available genres
        print("\nAvailable genres:")
        for i, genre in enumerate(available_genres, 1):
            print(f"{i}. {genre}")

        # Get and validate genre selection
        while True:
            genre_input = input("\nEnter a genre from the list: ")
            if genre_input in available_genres:
                break
            else:
                print("Invalid genre. Please select from the list.")

        # Get and validate number of recommendations
        while True:
            try:
                n = int(input("How many recommendations would you like? "))
                if n > 0:
                    return {"type": "genre", "genre": genre_input, "n": n}
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

def main():
    """
    Main function that orchestrates the movie recommendation system.

    This function:
    1. Loads the movie and ratings data
    2. Calculates popularity scores
    3. Gets user preferences
    4. Provides recommendations based on those preferences
    5. Offers to visualize the results
    """
    # Step 1: Load data
    print("Loading movie data...")
    movies_df, ratings_df = load_data()

    # Check if data was loaded successfully
    if movies_df is None or ratings_df is None:
        print("Failed to load data. Exiting.")
        return

    # Step 2: Calculate popularity scores
    print("Calculating movie popularity...")
    popular_movies = calculate_popularity(movies_df, ratings_df)

    # Step 3: Extract available genres
    available_genres = get_available_genres(movies_df)

    # Step 4: Get user preferences
    user_prefs = get_user_input(available_genres)

    # Step 5: Provide recommendations based on user input
    if user_prefs["type"] == "popularity":
        # Display popularity-based recommendations
        print(f"\nTop {user_prefs['n']} Popular Movies:")
        top_movies = recommend_popular_movies(popular_movies, n=user_prefs['n'])
        for i, (_, movie) in enumerate(top_movies.iterrows(), 1):
            print(f"{i}. {movie['title']} - Popularity Score: {movie['popularity_score']:.2f}")
    else:
        # Display genre-based recommendations
        print(f"\nTop {user_prefs['n']} Popular {user_prefs['genre']} Movies:")
        genre_movies = recommend_popular_movies(popular_movies, n=user_prefs['n'], genre=user_prefs['genre'])
        if genre_movies is not None:
            for i, (_, movie) in enumerate(genre_movies.iterrows(), 1):
                print(f"{i}. {movie['title']} - Popularity Score: {movie['popularity_score']:.2f}")

    # Step 6: Offer visualization option
    visualize = input("\nWould you like to visualize the results? (y/n): ").lower()
    if visualize == 'y':
        print("\nCreating visualization...")
        if user_prefs["type"] == "popularity":
            visualize_popularity(popular_movies)
        else:
            visualize_popularity(popular_movies, genre=user_prefs['genre'])

    print("\nMovie recommendation system completed!")

# Entry point of the program
if __name__ == "__main__":
    main()