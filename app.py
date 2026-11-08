import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Page Configuration
st.set_page_config(page_title="Semantic Movie Genre Tagger", layout="centered")
st.title("🎬 Semantic Movie Genre Tagger")
st.write("Powered by HuggingFace MiniLM and a Multi-Layer Perceptron")

# 2. Cache the heavy lifting
# @st.cache_resource prevents Streamlit from reloading the 384-dimensional 
# transformer model into RAM every time a user clicks a button.
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    mlp = joblib.load('models/movie_genre_mlp_tuned.pkl')
    mlb = joblib.load('models/genre_label_encoder.pkl')
    return embedder, mlp, mlb

with st.spinner("Loading semantic embedding space and neural network..."):
    embedder, mlp, mlb = load_models()

# 3. User Interface
st.markdown("### Enter a Movie Plot Summary")
user_plot = st.text_area("Plot Summary", height=200, placeholder="A lonely astronaut struggles with isolation...")
submit_button = st.button("Predict Genres")

# 4. Inference Pipeline
if submit_button and user_plot:
    with st.spinner("Analyzing semantic context..."):
        # Step A: Convert text to dense vector
        plot_embedding = embedder.encode([user_plot])
        
        # Step B: Extract raw probabilities from the hidden layers
        probabilities = mlp.predict_proba(plot_embedding)[0]
        
        # Step C: Apply our optimized threshold of 0.25
        # This leverages the high recall we established in Colab
        threshold = 0.25
        active_indices = np.where(probabilities >= threshold)[0]
        
        # Step D: Map active mathematical indices back to text labels
        if len(active_indices) > 0:
            predicted_genres = mlb.classes_[active_indices]
            
            st.success("### Predicted Genres:")
            # Display genres as clean pills/tags
            cols = st.columns(len(predicted_genres))
            for i, genre in enumerate(predicted_genres):
                with cols[i]:
                    st.button(genre, key=genre)
        else:
            st.warning("The neural network could not confidently identify a genre at the current threshold.")