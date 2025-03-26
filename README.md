# Movie Recommendation System Based on Popularity

This project implements a simple movie recommendation system that suggests movies based on their popularity. It uses the MovieLens dataset to calculate popularity scores and recommend movies.

## Features

- Recommends movies based on popularity metrics
- Popularity is calculated using a weighted combination of:
  - Average user ratings
  - Number of ratings received
- Can filter recommendations by genre
- Visualizes top popular movies

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib
- scikit-learn

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

Run the script:

```bash
python movie_recommender.py
```

The script will:
1. Download the MovieLens dataset (if not already present)
2. Calculate popularity scores for all movies
3. Display the top 10 most popular movies
4. Display the top 5 most popular action movies
5. Create a visualization of the top 20 popular movies

## How It Works

### Data Loading
The system uses the MovieLens dataset, which contains movie information and user ratings. If the data isn't available locally, it downloads it from GitHub.

### Popularity Calculation
Popularity is calculated using:
- Average rating: How highly users rated the movie
- Number of ratings: How many users rated the movie

These metrics are normalized and combined with weights to create a single popularity score.

### Recommendation
Movies are ranked by their popularity score, and the top N movies are recommended. You can also filter by genre to get genre-specific recommendations.

## Extending the System

This is a basic recommendation system. To improve it, you could:

1. Add more features to the popularity calculation (e.g., recency of ratings)
2. Implement content-based filtering using movie attributes
3. Implement collaborative filtering to provide personalized recommendations
4. Add a user interface for easier interaction

## License

This project is open source and available under the MIT License.