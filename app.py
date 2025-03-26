import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from sklearn.feature_extraction.text import TfidfVectorizer



class CustomTFIDFEmbeddings(Embeddings):
    """
    Custom Embeddings class using TF-IDF vectorization
    """
    def __init__(self, max_features=1000):
        """
        Initialize TF-IDF Vectorizer
        
        Args:
            max_features (int): Maximum number of features
        """
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

    def fit(self, texts):
        """
        Fit the vectorizer to the given texts
        
        Args:
            texts (List[str]): List of texts to fit
        """
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed_documents(self, texts):
        """
        Embed a list of documents
        
        Args:
            texts (List[str]): List of text documents to embed
        
        Returns:
            List[List[float]]: List of embeddings
        """
        if not self._fitted:
            self.fit(texts)
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text):
        """
        Embed a query text
        
        Args:
            text (str): Text to embed
        
        Returns:
            List[float]: Embedding of the text
        """
        if not self._fitted:
            raise ValueError("Embeddings must be fitted before querying")
        return self.vectorizer.transform([text]).toarray()[0].tolist()

class PDFChatbot:
    def __init__(self, pdf_paths, groq_api_key):
        """
        Initialize PDF Chatbot with Groq Llama model
        
        Args:
            pdf_paths (list): List of PDF file paths
            groq_api_key (str): Groq API key
        """
        # Set Groq API Key
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        # Load and process PDFs
        self.retriever = self._load_multiple_pdfs(pdf_paths)
        
        # Initialize Groq Llama model
        self.llm = ChatGroq(
            temperature=0.2,  # Low temperature for more focused responses
            model_name="llama2-70b-4096"  # Groq's Llama 2 70B model
        )
        
        # Create QA chain
        if self.retriever is not None:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm, 
                chain_type="stuff", 
                retriever=self.retriever,
                return_source_documents=True
            )
        else:
            self.qa_chain = None
            print("Warning: No retriever created. Chatbot may not function correctly.")
    
    def _load_multiple_pdfs(self, pdf_paths):
        """
        Load and process multiple PDF files
        
        Args:
            pdf_paths (list): List of PDF file paths
        
        Returns:
            Retriever object or None
        """
        try:
            all_documents = []
            all_texts = []

            for pdf_path in pdf_paths:
                if not os.path.exists(pdf_path):
                    print(f"Warning: PDF file {pdf_path} not found!")
                    continue
                
                loader = PyMuPDFLoader(pdf_path)
                documents = loader.load()
                all_documents.extend(documents)
                
                # Extract text for embeddings
                all_texts.extend([doc.page_content for doc in documents])

            # Check if any documents were loaded
            if not all_documents:
                raise ValueError("No documents could be loaded from the PDF files")

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            docs = text_splitter.split_documents(all_documents)

            # Use TF-IDF embeddings
            embeddings = CustomTFIDFEmbeddings()
            
            # Fit embeddings on all text
            embeddings.fit(all_texts)

            # Store embeddings in FAISS
            vectorstore = FAISS.from_documents(docs, embeddings)

            return vectorstore.as_retriever()
        
        except Exception as e:
            print(f"Error loading PDFs: {e}")
            return None

# Initialize Flask App
app = Flask(__name__)

CORS(app)

# Configuration (use environment variables in production)
GROQ_API_KEY = ""
PDF_FILES = ["pdf1.pdf"]

# Create Chatbot Instance
try:
    chatbot = PDFChatbot(PDF_FILES, GROQ_API_KEY)
except Exception as e:
    print(f"Failed to initialize chatbot: {e}")
    chatbot = None

@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint to handle user questions
    """
    # Check if chatbot is initialized
    if chatbot is None or chatbot.qa_chain is None:
        return jsonify({"error": "Chatbot not properly initialized"}), 500

    try:
        # Get question from request
        data = request.json
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400

        question = data['question']
        
        # Run the query
        result = chatbot.qa_chain({"query": question})
        
        return jsonify({
            "answer": result['result'],
            "source_documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in result.get('source_documents', [])
            ]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy", 
        "model": "Groq Llama2-70B",
        "pdf_files_loaded": len(PDF_FILES),
        "chatbot_initialized": chatbot is not None
    }), 200

if __name__ == '__main__':
    app.run(debug=True)