import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---------------------------
# Step 1: Load the Dataset
# ---------------------------
# Replace with your actual file name if needed
df = pd.read_csv('IMDb_Movies_India.csv', encoding='latin1', engine='python', on_bad_lines='skip')

# ---------------------------
# Step 2: Drop Missing Values
# ---------------------------
df = df.dropna()

# ---------------------------
# Step 3: Feature Selection
# ---------------------------
features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
target = 'Rating'

# One-hot encode categorical variables
X = pd.get_dummies(df[features])
y = df[target]

# ---------------------------
# Step 4: Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Step 5: Train the Model
# ---------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------
# Step 6: Evaluate the Model
# ---------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# ---------------------------
# Step 7: Prediction Function
# ---------------------------
def predict_movie_rating(genre, director, actor1, actor2, actor3):
    """
    Predict the rating of a new movie given its key features.
    """
    input_dict = {
        'Genre': [genre],
        'Director': [director],
        'Actor 1': [actor1],
        'Actor 2': [actor2],
        'Actor 3': [actor3]
    }
    input_df = pd.DataFrame(input_dict)
    input_encoded = pd.get_dummies(input_df)
    input_aligned = input_encoded.reindex(columns=X.columns, fill_value=0)

    predicted_rating = model.predict(input_aligned)[0]
    return predicted_rating

# ---------------------------
# Step 8: Example Usage
# ---------------------------
example_prediction = predict_movie_rating(
    genre='Action',
    director='Christopher Nolan',
    actor1='Christian Bale',
    actor2='Tom Hardy',
    actor3='Anne Hathaway'
)

print("Predicted Rating for Example Movie:", round(example_prediction, 2))
