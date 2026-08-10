from langchain_ollama import OllamaEmbeddings
from src.tratamento.loading import obter_pedacos

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)

if __name__ == "__main__":
    pedacos = obter_pedacos()
    print(embeddings_model.embed_query(pedacos[0].page_content)) # verificar o resultado do embedding do primeiro pedaço.
