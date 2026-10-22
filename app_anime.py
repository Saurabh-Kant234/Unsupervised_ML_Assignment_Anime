import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title='Anime Recommender App', page_icon='🎬', layout='wide'
)

st.title('🎬 Content-Based Anime Recommender System')
st.markdown(
    'Select an anime below to get personalized content-based recommendations!'
)


@st.cache_resource
def load_anime_artifacts():
  cosine_sim = joblib.load('cosine_sim.pkl')
  anime_df = joblib.load('anime_df.pkl')
  return cosine_sim, anime_df


try:
  cosine_sim, anime_df = load_anime_artifacts()

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
  st.error(
      f'Error loading model files: {e}. Make sure `cosine_sim.pkl` and'
      ' `anime_df.pkl` are in the same folder.'
  )