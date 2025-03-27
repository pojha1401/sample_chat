import os
import pickle
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def chunk_text(text, chunk_size=2000):
    """Divide text into chunks of specified size."""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2", cache_dir="embeddings_cache"):
    """
    Generate vector embeddings for each text chunk with caching mechanism.
    
    Args:
        chunks (list): List of text chunks
        model_name (str): Name of the embedding model
        cache_dir (str): Directory to store cached embeddings
    
    Returns:
        tuple: (embeddings, chunks)
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate a unique cache filename based on the chunks
    import hashlib
    chunks_hash = hashlib.md5(''.join(chunks).encode()).hexdigest()
    cache_filename = os.path.join(cache_dir, f"{model_name}_{chunks_hash}_embeddings.pkl")
    
    # Check if cached embeddings exist
    if os.path.exists(cache_filename):
        print("Loading embeddings from cache...")
        with open(cache_filename, 'rb') as f:
            return pickle.load(f)
    
    # If no cache, generate embeddings
    try:
        print("Generating embeddings...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks, convert_to_numpy=True)
        
        # Cache the embeddings
        with open(cache_filename, 'wb') as f:
            pickle.dump((embeddings, chunks), f)
        
        return embeddings, chunks
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None, None

def retrieve_relevant_chunks_knn(query, chunks, embeddings, k=5, model_name="all-MiniLM-L6-v2"):
    """
    Retrieve the k most relevant chunks using K-Nearest Neighbors approach.
    
    Args:
        query (str): Search query
        chunks (list): Original text chunks
        embeddings (numpy.ndarray): Pre-computed embeddings
        k (int): Number of nearest neighbors to retrieve
        model_name (str): Embedding model name
    
    Returns:
        list: K most relevant chunks
    """
    try:
        model = SentenceTransformer(model_name)
        query_embedding = model.encode(query, convert_to_numpy=True).reshape(1, -1)
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Get indices of top k most similar chunks
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        # Return the corresponding chunks
        return [chunks[idx] for idx in top_k_indices]
    except Exception as e:
        print(f"Error retrieving relevant chunks: {e}")
        return []

def query_llama_groq(prompt, model_name="llama3-70b-8192", api_key="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"):
    """Query LLaMA-3 using Groq API."""
    client = Groq(api_key=api_key)

    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Synthesize information from multiple context chunks to provide a comprehensive answer. Use the context to provide a detailed answer.",
        },
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    pdf_path = "pdf1.pdf"  # Change this to your actual PDF file path
    extracted_text = extract_text_from_pdf(pdf_path)
    text_chunks = chunk_text(extracted_text)
    
    # Generate embeddings
    chunk_embeddings, chunks = generate_embeddings(text_chunks)
    
    if chunk_embeddings is not None:
        query = "Give me information in the document about karnataka"
        
        # Retrieve top 5 most relevant chunks
        relevant_chunks = retrieve_relevant_chunks_knn(query, chunks, chunk_embeddings, k=5)
        
        # Combine chunks into a single context
        combined_context = " ".join(relevant_chunks)
        
        # Generate response using combined context
        llm_response = query_llama_groq(f"Based on these context chunks: {combined_context}, answer: {query}")
        print(f"LLM Response:\n{llm_response}\n")