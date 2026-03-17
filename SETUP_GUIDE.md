# Social Media Friend Recommendation System - Setup Guide

## Overview
This project implements a machine learning-based friend recommendation system for social media platforms. It uses cosine similarity to find users with similar interests, demographics, and location preferences.

## Features
- **Interest-based matching**: Analyzes user interests using one-hot encoding
- **Demographic similarity**: Considers age, gender, and location
- **Cosine similarity algorithm**: Calculates user similarity scores
- **Personalized recommendations**: Generates top-N friend suggestions
- **Jupyter notebook interface**: Interactive data analysis and visualization
- **Python script testing**: Command-line testing capabilities

## Prerequisites
- Python 3.7 or higher
- pip (Python package installer)
- Git (for version control)

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/vedanth-raj/ADSA-PROJECT.git
cd ADSA-PROJECT
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Sample Dataset (Optional)
If you don't have the original Kaggle dataset, create a sample dataset:
```bash
python create_sample_dataset.py
```

### 4. Run the System

#### Option A: Using Jupyter Notebook (Recommended)
```bash
jupyter notebook
```
Then open `SocialMediaUsersRecommendation.ipynb` in your browser.

#### Option B: Using Python Script
```bash
python test_recommendation_system.py
```

## Project Structure
```
ADSA-PROJECT/
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
├── SocialMediaUsersRecommendation.ipynb   # Main Jupyter notebook
├── create_sample_dataset.py              # Sample data generator
├── test_recommendation_system.py         # Testing script
├── SocialMediaUsersDataset.csv           # Dataset (generated)
├── user_similarity_matrix.npy            # Cached similarity matrix
└── .gitignore                            # Git ignore rules
```

## How It Works

### 1. Data Preprocessing
- **Age calculation**: Converts date of birth to numerical age
- **Interest encoding**: One-hot encodes user interests
- **Demographic encoding**: Encodes gender and country information
- **Feature normalization**: Combines all features into a unified matrix

### 2. Similarity Calculation
- Uses **cosine similarity** to measure user similarity
- Considers multiple factors: interests, age, gender, location
- Generates a similarity matrix for all users

### 3. Recommendation Generation
- Finds users with highest similarity scores
- Excludes the target user from recommendations
- Returns top-N most similar users with details

## Usage Examples

### Getting Recommendations for User ID 1
```python
recommendations = get_friend_recommendations(
    user_id=1, 
    dataset=dataset, 
    similarity_matrix=similarity_matrix, 
    top_n=5
)
```

### Sample Output
```
User: Diana Prince 1
Age: 45, Gender: Male
Location: Houston, Denmark
Interests: 'Music', 'Finance and investments', 'Parenting and family'

Top 5 Friend Recommendations:
1. Diana Prince 162 (Similarity: 0.999)
2. Grace Lee 265 (Similarity: 0.999)
3. Frank Miller 705 (Similarity: 0.999)
```

## Dataset Information

### Required Columns
- **UserID**: Unique identifier for each user
- **Name**: User's name or username
- **Gender**: User's gender (Male/Female)
- **DOB**: Date of birth (YYYY-MM-DD format)
- **Interests**: Comma-separated list of interests
- **City**: User's city of residence
- **Country**: User's country

### Sample Data Format
```csv
UserID,Name,Gender,DOB,Interests,City,Country
1,Alice Johnson,Female,1990-05-15,"'Movies', 'Fashion', 'Books'",New York,United States
```

## Customization Options

### Adjusting Recommendation Parameters
- **top_n**: Number of recommendations to return
- **similarity_threshold**: Minimum similarity score for recommendations
- **feature_weights**: Adjust importance of different features

### Adding New Features
- **Education level**: Add educational background matching
- **Profession**: Include career-based similarity
- **Social connections**: Consider mutual friends
- **Activity patterns**: Analyze usage behavior

## Performance Considerations

### For Large Datasets (>10,000 users)
- Use **approximate similarity** algorithms (LSH, Annoy)
- Implement **batch processing** for similarity calculations
- Consider **dimensionality reduction** (PCA, t-SNE)
- Use **sparse matrices** for memory efficiency

### Optimization Tips
- Cache similarity matrices using `numpy.save()`
- Precompute recommendations for active users
- Use parallel processing for similarity calculations
- Implement incremental updates for new users

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
pip install --upgrade pandas scikit-learn numpy jupyter
```

#### 2. Dataset Not Found
```bash
python create_sample_dataset.py
```

#### 3. Memory Issues with Large Datasets
- Reduce dataset size or use sampling
- Implement batch processing
- Use sparse matrices

#### 4. Jupyter Notebook Won't Start
```bash
pip install --upgrade jupyter
jupyter notebook --generate-config
```

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License
This project is open source and available under the MIT License.

## Contact
For questions or support, please open an issue on GitHub or contact the project maintainer.

## Future Enhancements
- **Real-time recommendations**: WebSocket-based live updates
- **Machine learning models**: Deep learning for better accuracy
- **Graph-based algorithms**: Social network analysis
- **A/B testing framework**: Recommendation algorithm comparison
- **Privacy features**: Differential privacy implementation
- **Mobile app integration**: REST API development