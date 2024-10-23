#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load and prepare the data
data = pd.read_csv('House Data.csv')

# Convert numeric columns to float, replacing non-numeric values with NaN
numeric_columns = ['total_sqft', 'bath', 'balcony', 'price']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Remove rows with NaN values
data = data.dropna()

# Separate features and target
X = data.drop('price', axis=1)
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numeric and categorical columns
numeric_features = ['total_sqft', 'bath', 'balcony']
categorical_features = ['area_type', 'availability', 'location', 'size', 'society']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create and train the model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Corrected: model.fit, not model,fit
model.fit(X_train, y_train)

# Function to get user input
def get_user_input():
    return pd.DataFrame({
        'area_type': [input("Area Type (e.g., Super build-up Area): ")],
        'availability': [input("Availability (e.g., Ready to Move): ")],
        'location': [input("Location (e.g., Electronic city Phase II): ")],
        'size': [input("Size (e.g., 2BHK): ")],
        'society': [input("Society: ")],
        'total_sqft': [float(input("Total Square Feet : "))],
        'bath': [float(input("Number of Bathrooms : "))],
        'balcony': [float(input("Number of Balconies : "))]
    })

# Predict price for a new house
new_house = get_user_input()

# Ensure new input data goes through the preprocessor before making predictions
new_house_preprocessed = model.named_steps['preprocessor'].transform(new_house)

# Make the prediction
predicted_price = model.named_steps['regressor'].predict(new_house_preprocessed)
print(f"\nPredicted Price: {predicted_price[0]: .2f}")


# In[ ]:




