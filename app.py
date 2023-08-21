import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


df = pd.read_csv('cleaned_data.csv')




# load the vectorizers on the item features
description_vectors =  pickle.load(open('description_vectors.pkl', 'rb'))
category_vectors = pickle.load(open('category_vectors.pkl', 'rb'))
location_vectors = pickle.load(open('location_vectors.pkl', 'rb'))
vectorizer_desc =  pickle.load(open('vectorizer_desc.pkl', 'rb'))
vectorizer_cat = pickle.load(open('vectorizer_cat.pkl', 'rb'))
vectorizer_loc = pickle.load(open('vectorizer_loc.pkl', 'rb'))


@st.cache
def get_recommendations(user_id, top_n=5):
    # Filter data based on user_id
    user_orders = df[df['User_id'] == user_id]

    # Create a user profile by combining the descriptions of their previous orders
    user_profile_desc = ' '.join(user_orders['Food_description'])
    user_profile_cat = ' '.join(user_orders['Food_categories'])
    user_profile_loc = ' '.join(user_orders['restaurant_location'])
    
    # Vectorize the user profile
    user_profile_desc_vector = vectorizer_desc.transform([user_profile_desc])
    user_profile_cat_vector = vectorizer_cat.transform([user_profile_cat])
    user_profile_loc_vector = vectorizer_loc.transform([user_profile_loc])
  
    # Calculate the similarity between user profile and food items
    description_similarity_scores = linear_kernel(user_profile_desc_vector, description_vectors)[0]
    category_similarity_scores = linear_kernel(user_profile_cat_vector, category_vectors)[0]
    location_similarity_scores = linear_kernel(user_profile_loc_vector, location_vectors)[0]

    # Calculate the weighted average of similarities
    similarity_scores = (
        0.8 * description_similarity_scores
        + 0.1 * category_similarity_scores
        + 0.1 * location_similarity_scores
    )

    # Sort by similarity scores and get the top recommendations
    top_indices = similarity_scores.argsort()[::-1][:top_n]
    recommendations = df.loc[top_indices, ['Food_name','Restaurant']]

    return recommendations

def main():
    st.title('Recommender System')

    # Create input fields for user ID and number of recommendations
    user_id = st.text_input('User ID')
    top_n = st.number_input('Number of Recommendations', min_value=1, max_value=10, value=5)

    # Generate recommendations when the user clicks the button
    if st.button('Generate Recommendations'):
        recommendations = get_recommendations(user_id, top_n)
        st.write(recommendations)

if __name__ == '__main__':
    main()
