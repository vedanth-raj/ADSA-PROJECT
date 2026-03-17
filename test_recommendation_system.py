import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import numpy as np

def calculate_age(dob_str):
    """Calculate age from date of birth string"""
    try:
        dob = datetime.strptime(dob_str, '%Y-%m-%d')
        today = datetime.now()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return float(age)
    except:
        return 25.0  # Default age if parsing fails

def preprocess_interests(interests_str):
    """Clean and preprocess interests string"""
    # Remove quotes and split by comma
    interests = interests_str.replace("'", "").split(", ")
    return interests

def get_friend_recommendations(user_id, dataset, similarity_matrix, top_n=5):
    """Get friend recommendations for a specific user"""
    user_idx = user_id - 1  # Convert to 0-based index
    
    if user_idx >= len(dataset):
        return f"User ID {user_id} not found in dataset"
    
    # Get similarity scores for the user
    user_similarities = similarity_matrix[user_idx]
    
    # Get indices of most similar users (excluding the user themselves)
    similar_indices = np.argsort(user_similarities)[::-1][1:top_n+1]
    
    recommendations = []
    for idx in similar_indices:
        similar_user = dataset.iloc[idx]
        similarity_score = user_similarities[idx]
        recommendations.append({
            'UserID': similar_user['UserID'],
            'Name': similar_user['Name'],
            'Similarity Score': round(similarity_score, 3),
            'Common Interests': similar_user['Interests'],
            'Location': f"{similar_user['City']}, {similar_user['Country']}"
        })
    
    return recommendations

def main():
    print("Loading Social Media Friend Recommendation System...")
    
    # Load the dataset
    try:
        dataset = pd.read_csv('SocialMediaUsersDataset.csv')
        print(f"Dataset loaded successfully with {len(dataset)} users")
    except FileNotFoundError:
        print("Error: SocialMediaUsersDataset.csv not found. Please run create_sample_dataset.py first.")
        return
    
    # Feature extraction - Interests
    print("Extracting features...")
    
    # One-hot encode interests
    interests = dataset['Interests'].str.get_dummies(', ')
    interests.fillna(0, inplace=True)
    
    # Calculate age and add as feature
    dataset['Age'] = dataset['DOB'].apply(calculate_age)
    
    # One-hot encode gender
    gender_encoded = pd.get_dummies(dataset['Gender'], prefix='Gender')
    
    # One-hot encode country (limit to top countries to avoid too many features)
    top_countries = dataset['Country'].value_counts().head(10).index
    dataset['Country_Top'] = dataset['Country'].apply(lambda x: x if x in top_countries else 'Other')
    country_encoded = pd.get_dummies(dataset['Country_Top'], prefix='Country')
    
    # Combine all features
    features = pd.concat([
        interests,
        dataset[['Age']],
        gender_encoded,
        country_encoded
    ], axis=1)
    
    print(f"Feature matrix created with shape: {features.shape}")
    
    # Calculate similarity matrix
    print("Calculating user similarities...")
    similarity_matrix = cosine_similarity(features)
    print("Similarity matrix calculated successfully!")
    
    # Test the recommendation system
    print("\n" + "="*60)
    print("TESTING FRIEND RECOMMENDATION SYSTEM")
    print("="*60)
    
    # Show sample users
    print("\nSample users in the dataset:")
    print(dataset[['UserID', 'Name', 'Gender', 'Age', 'Interests', 'City', 'Country']].head(10))
    
    # Get recommendations for a few users
    test_users = [1, 5, 10]
    
    for user_id in test_users:
        print(f"\n{'='*50}")
        print(f"RECOMMENDATIONS FOR USER {user_id}")
        print(f"{'='*50}")
        
        # Show user details
        user_info = dataset[dataset['UserID'] == user_id].iloc[0]
        print(f"User: {user_info['Name']}")
        print(f"Age: {int(user_info['Age'])}")
        print(f"Gender: {user_info['Gender']}")
        print(f"Location: {user_info['City']}, {user_info['Country']}")
        print(f"Interests: {user_info['Interests']}")
        
        # Get recommendations
        recommendations = get_friend_recommendations(user_id, dataset, similarity_matrix, top_n=5)
        
        print(f"\nTop 5 Friend Recommendations:")
        print("-" * 50)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['Name']} (ID: {rec['UserID']})")
            print(f"   Similarity Score: {rec['Similarity Score']}")
            print(f"   Location: {rec['Location']}")
            print(f"   Interests: {rec['Common Interests']}")
            print()
    
    print("\n" + "="*60)
    print("RECOMMENDATION SYSTEM TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Save similarity matrix for future use
    np.save('user_similarity_matrix.npy', similarity_matrix)
    print("\nSimilarity matrix saved as 'user_similarity_matrix.npy'")

if __name__ == "__main__":
    main()