import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.set_page_config(
    page_title='Anime Recommender App', page_icon='🎬', layout='wide'
)

st.title('🎬 Content-Based Anime Recommender System')
st.markdown(
    'Select an anime below to get personalized content-based recommendations!'
)


# Cache resource ensures this heavy computation only runs ONCE when the app starts
@st.cache_resource
def load_data_and_similarity():
  # Load dataset directly
  anime_df = pd.read_csv('anime.csv')

  # Clean data
  anime_df['genre'] = anime_df['genre'].fillna('').astype(str)
  anime_df['type'] = anime_df['type'].fillna('').astype(str)
  anime_df['name'] = anime_df['name'].fillna('').astype(str)

  # Create combined features
  anime_df['features'] = (
      anime_df['genre'] + ' ' + anime_df['type'] + ' ' + anime_df['name']
  )

  # Compute TF-IDF and Cosine Similarity in-memory
  tfidf = TfidfVectorizer(stop_words='english')
  tfidf_matrix = tfidf.fit_transform(anime_df['features'])
  cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

  return anime_df, cosine_sim


try:
  # Load dataframe and similarity matrix
  anime_df, cosine_sim = load_data_and_similarity()

  # Dropdown to select/search anime
  anime_list = anime_df['name'].dropna().unique()
  selected_anime = st.selectbox('Select or Search an Anime:', anime_list)

  if st.button('Get Recommendations'):
    indices = pd.Series(anime_df.index, index=anime_df['name']).drop_duplicates()

    if selected_anime in indices:
      idx = indices[selected_anime]
      sim_scores = sorted(
          list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True
      )[1:11]
      recommended_names = [anime_df['name'].iloc[i[0]] for i in sim_scores]

      st.success(f'Top 10 Recommendations for **{selected_anime}**:')
      for i, name in enumerate(recommended_names, 1):
        st.write(f'{i}. {name}')
    else:
      st.error('Selected anime not found in database.')

except Exception as e:
  st.error(f'Error loading application: {e}')