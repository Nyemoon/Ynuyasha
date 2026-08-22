import httpx
import os
from langchain_ollama import OllamaEmbeddings

from src.tratamento.loading import obter_pedacos

# Configura o pool HTTP do OllamaEmbeddings via sync_client_kwargs (suportado
# por esta versão do langchain-ollama). Um timeout generoso evita cortes em
# gerações longas, e o pool com keep-alive reduz a latência por query.
embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
    keep_alive=1800,
    client_kwargs={
        "timeout": httpx.Timeout(900.0, connect=30.0, read=900.0, write=900.0, pool=30.0),
        "limits": httpx.Limits(max_connections=10, max_keepalive_connections=5),
    },
)


def aquecer_embeddings() -> None:
    """Pré-carrega o modelo de embedding no Ollama (warm-up).

    A primeira chamada de embed_query demora muito mais (carrega o modelo no
    servidor). Chamar isto no boot reduz a latência da primeira pergunta do RAG.
    Failure é silenciosa: se o Ollama estiver fora do ar, apenas não aquecido.
    """
    try:
        embeddings_model.embed_query("warm-up")
    except Exception:
        pass

if __name__ == "__main__":
    pedacos = obter_pedacos()
    print(embeddings_model.embed_query(pedacos[0].page_content)) # verificar o resultado do embedding do primeiro pedaço.
