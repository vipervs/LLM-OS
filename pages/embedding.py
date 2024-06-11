from langchain_community.embeddings import OllamaEmbeddings

def get_embedding(text):
    print(f"Requesting embedding for: {text}")
    try:
        embeddings = OllamaEmbeddings(model="snowflake-arctic-embed:latest")
        return embeddings.embed_query(text)
    except Exception as e:
        print(f"Error requesting embedding: {e}")
        return None
