import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# -------------------------
# Config
# -------------------------
INDEX_NAME = "techcrunch-articles"  
EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
LLM_MODEL = "openai/gpt-oss-20b" 
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


# -------------------------
# Keys
# -------------------------
load_dotenv()  
nvidia_key = os.getenv("nvidia_key")
pinecone_key = os.getenv("pinecone_key")


# -------------------------
# Query side (LangChain)
# -------------------------
pc = Pinecone(api_key=pinecone_key)
index = pc.Index(INDEX_NAME)
print(index)

embedder = NVIDIAEmbeddings(
    model=EMBED_MODEL,
    base_url=NVIDIA_BASE_URL,
    api_key=nvidia_key
)

llm = ChatNVIDIA(
    model=LLM_MODEL,
    base_url=NVIDIA_BASE_URL,
    api_key=nvidia_key
)

vector_store = PineconeVectorStore(
    index=index,        
    embedding=embedder,
    text_key="text"    
)

query = "Tell me what does Meta violate recently"
docs = vector_store.similarity_search(query, k=5)

context = "\n\n".join(d.page_content for d in docs)
prompt = PromptTemplate.from_template(
    "You are an assistant that answers based ONLY on the context below.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

answer = (prompt | llm).invoke({"question": query, "context": context})
print("\n=== Answer ===\n", answer)

for i, d in enumerate(docs, 1):
    print(f"\n--- Doc {i} ---")
    print(d.page_content[:1000])  # preview first 1000 chars
    print("Metadata:", d.metadata)