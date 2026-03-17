import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Sample data for creating a demo dataset
names = [
    "Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Prince", "Eve Adams",
    "Frank Miller", "Grace Lee", "Henry Wilson", "Ivy Chen", "Jack Davis",
    "Kate Thompson", "Liam Garcia", "Mia Rodriguez", "Noah Martinez", "Olivia Anderson",
    "Paul Taylor", "Quinn Moore", "Ruby Jackson", "Sam White", "Tina Harris"
]

genders = ["Male", "Female"]
interests_list = [
    "Movies", "Fashion", "Books", "Gaming", "Finance and investments", 
    "Outdoor activities", "DIY and crafts", "Music", "Science", "Cars and automobiles",
    "Politics", "History", "Pets", "Fitness", "Sports", "Technology", "Travel",
    "Cooking", "Art", "Photography", "Beauty", "Education and learning",
    "Business and entrepreneurship", "Parenting and family", "Social causes and activism",
    "Gardening"
]

cities = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
    "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
    "Fort Worth", "Columbus", "Charlotte", "San Francisco", "Indianapolis",
    "Seattle", "Denver", "Washington"
]

countries = [
    "United States", "Canada", "United Kingdom", "Germany", "France", "Italy",
    "Spain", "Australia", "Japan", "South Korea", "India", "Brazil", "Mexico",
    "Argentina", "Netherlands", "Sweden", "Norway", "Denmark", "Finland", "Switzerland"
]

# Generate sample data
data = []
for i in range(1000):  # Create 1000 sample users
    user_id = i + 1
    name = random.choice(names) + f" {i+1}"
    gender = random.choice(genders)
    
    # Generate random birth date (age between 18-70)
    start_date = datetime.now() - timedelta(days=70*365)
    end_date = datetime.now() - timedelta(days=18*365)
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    dob = random_date.strftime("%Y-%m-%d")
    
    # Generate random interests (2-5 interests per user)
    num_interests = random.randint(2, 5)
    user_interests = random.sample(interests_list, num_interests)
    interests_str = "'" + "', '".join(user_interests) + "'"
    
    city = random.choice(cities)
    country = random.choice(countries)
    
    data.append([user_id, name, gender, dob, interests_str, city, country])

# Create DataFrame
df = pd.DataFrame(data, columns=["UserID", "Name", "Gender", "DOB", "Interests", "City", "Country"])

# Save to CSV
df.to_csv("SocialMediaUsersDataset.csv", index=False)
print(f"Sample dataset created with {len(df)} users")
print("First 5 rows:")
print(df.head())