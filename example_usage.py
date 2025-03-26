"""
Example usage of the movie recommendation system.
This script demonstrates how to use the movie recommender with custom parameters.
"""

from movie_recommender_new import load_data, calculate_popularity, recommend_popular_movies, visualize_popularity

def main():
    print("Loading movie data...")
    movies_df, ratings_df = load_data()
    
    if movies_df is None or ratings_df is None:
        print("Failed to load data. Exiting.")
        return
    
    print("Calculating movie popularity...")
    popular_movies = calculate_popularity(movies_df, ratings_df)
    
    print("\nExample 1: Top 15 Popular Movies")
    top_15 = recommend_popular_movies(popular_movies, n=15)
    for i, (_, movie) in enumerate(top_15.iterrows(), 1):
        print(f"{i}. {movie['title']} - Popularity Score: {movie['popularity_score']:.2f}")
    
    print("\nExample 2: Top 5 Popular Comedy Movies")
    comedy_movies = recommend_popular_movies(popular_movies, n=5, genre='Comedy')
    if comedy_movies is not None:
        for i, (_, movie) in enumerate(comedy_movies.iterrows(), 1):
            print(f"{i}. {movie['title']} - Popularity Score: {movie['popularity_score']:.2f}")
    
    print("\nExample 3: Top 5 Popular Sci-Fi Movies")
    scifi_movies = recommend_popular_movies(popular_movies, n=5, genre='Sci-Fi')
    if scifi_movies is not None:
        for i, (_, movie) in enumerate(scifi_movies.iterrows(), 1):
            print(f"{i}. {movie['title']} - Popularity Score: {movie['popularity_score']:.2f}")
    
    print("\nExample 4: Visualizing Top 10 Popular Movies")
    visualize_popularity(popular_movies, top_n=10)
    
    print("\nExamples completed!")

if __name__ == "__main__":
    main()