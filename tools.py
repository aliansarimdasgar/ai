import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load API key
API_KEY = os.getenv("GOOGLE_API_KEY")

# Function to build Vector Store from CSV
def build_vector_store(csv_file):
    if not os.path.exists(csv_file):
        return None

    loader = CSVLoader(file_path=csv_file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, embedding_model)
    vector_store.save_local("faiss_index")
    return vector_store

# Load or build vector store
def load_or_build_vector_store():
    if not os.path.exists("faiss_index"):
        return build_vector_store("data.csv")  # Update with actual CSV path
    else:
        return FAISS.load_local(
            "faiss_index",
            GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
            allow_dangerous_deserialization=True
        )
