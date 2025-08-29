import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample food data (you can replace it with your own dataset)
data = {
    'Food': ['Pizza', 'Pasta', 'Burger', 'Sushi', 'Salad', 'Tacos', 'Pancakes', 'Sushi Roll', 'Lasagna'],
    'Ingredients': [
        'dough, cheese, tomato, pepperoni',
        'pasta, tomato sauce, garlic, cheese',
        'beef, lettuce, tomato, cheese, bun',
        'fish, rice, seaweed, wasabi, soy sauce',
        'lettuce, tomato, cucumber, dressing',
        'corn, meat, lettuce, cheese, salsa',
        'flour, eggs, milk, sugar, syrup',
        'rice, fish, avocado, seaweed, soy sauce',
        'pasta, tomato sauce, cheese, meat'
    ],
    'Category': ['Italian', 'Italian', 'American', 'Japanese', 'Healthy', 'Mexican', 'Breakfast', 'Japanese', 'Italian']
}

# Create DataFrame
df = pd.DataFrame(data)

# Function to get food recommendations based on a given food item
def recommend_food(input_food):
    # Initialize TF-IDF Vectorizer to transform ingredients into vectors
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Ingredients'])

    # Calculate cosine similarity between input food and all other foods
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Find index of input food
    idx = df[df['Food'] == input_food].index[0]

    # Get the pairwise similarity scores of all foods with the input food
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the foods based on similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 5 most similar foods
    sim_scores = sim_scores[1:6]

    # Get the food indices and names
    food_indices = [i[0] for i in sim_scores]
    recommended_foods = df['Food'].iloc[food_indices].tolist()

    return recommended_foods

# Example of getting recommendations for 'Pizza'
input_food = 'Pasta'
recommended_foods = recommend_food(input_food)

print(f"Food recommendations similar to {input_food}:")
for food in recommended_foods:
    print(f"- {food}")