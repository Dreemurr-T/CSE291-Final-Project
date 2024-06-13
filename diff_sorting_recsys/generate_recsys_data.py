import pandas as pd
import numpy as np


def generate_sample_data(num_users, num_items, num_ratings, filename):
    assert num_ratings <= num_users * num_items, "Number of ratings exceeds the maximum possible pairs of users and items."
    
    users = np.random.choice(num_users, num_ratings, replace=True)
    items = np.random.choice(num_items, num_ratings, replace=True)
    ratings = np.random.randint(1, 6, size=num_ratings)  # Ratings between 1 and 5
    
    data = {
        'user_id': users,
        'item_id': items,
        'rating': ratings
    }
    df = pd.DataFrame(data)
    
    # Drop duplicates to ensure unique user-item pairs
    df = df.drop_duplicates(subset=['user_id', 'item_id'])
    
    # If we have less data due to duplicates being dropped, fill it up again
    while len(df) < num_ratings:
        additional_users = np.random.choice(num_users, num_ratings - len(df), replace=True)
        additional_items = np.random.choice(num_items, num_ratings - len(df), replace=True)
        additional_ratings = np.random.randint(1, 6, size=num_ratings - len(df))
        additional_data = pd.DataFrame({
            'user_id': additional_users,
            'item_id': additional_items,
            'rating': additional_ratings
        })
        additional_data = additional_data.drop_duplicates(subset=['user_id', 'item_id'])
        df = pd.concat([df, additional_data]).drop_duplicates(subset=['user_id', 'item_id'])
    
    df.to_csv(filename, index=False)
    print(f"Sample data saved to {filename}")

# Generate and save the data
generate_sample_data(1000, 1000, 30000, 'sample_data.csv')