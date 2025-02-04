import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

st.title('Embedding Similarity App')

uploaded_file = st.file_uploader("Upload a NumPy dictionary of embeddings (.npy)", type=["npy"])

if uploaded_file is not None:
    # Load the .npy file as a dictionary
    doc_embeddings = np.load(BytesIO(uploaded_file.read()), allow_pickle=True).item()  # Convert to dictionary

    # Convert dictionary values to a NumPy array
    doc_ids = list(doc_embeddings.keys())  # List of document IDs
    embeddings = np.array(list(doc_embeddings.values()))  # Extract the actual embeddings

    # Display the shape of the embeddings
    st.write(f"Loaded {len(doc_embeddings)} document embeddings, each of size {embeddings.shape[1]}")

    # Define the list of models (for demonstration purposes)
    models = ['Model A', 'Model B', 'Model C']
    selected_model = st.selectbox('Select a model:', models)

    # Create an input box for user text input
    user_input = st.text_input('Enter your text:')

    if st.button('Submit'):
        # Placeholder for converting user input to embeddings
        # Replace this with actual model prediction logic
        user_embedding = np.random.rand(1, embeddings.shape[1])  # Dummy embedding

        # Calculate cosine similarity
        similarities = cosine_similarity(user_embedding, embeddings)

        # Get the top-k most similar document indexes
        top_k = 5
        top_k_indexes = np.argsort(similarities[0])[-top_k:][::-1]

        # Retrieve document IDs based on the indexes
        top_k_doc_ids = [doc_ids[i] for i in top_k_indexes]

        # Display the top-k most similar document IDs
        st.write('Top-k most similar document IDs:', top_k_doc_ids)
        ##
