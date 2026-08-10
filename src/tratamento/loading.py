from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parents[2]  # raiz do projeto: agente_Ynuyasha
DATASET_DIR = BASE_DIR / "data" / "dataset"


def carregar_documentos(diretorio: Path = DATASET_DIR) -> list:
    """Carrega todos os CSVs do diretório como documentos (uma linha por documento)."""
    carregador = DirectoryLoader(
        str(diretorio),
        glob="**/*.csv",
        loader_cls=CSVLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return carregador.load()


def dividir_em_pedacos(documentos: list) -> list:
    """Quebra os documentos em pedaços menores para o RAG."""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    return text_splitter.split_documents(documentos)


def obter_pedacos() -> list:
    """Carrega os documentos e devolve os pedaços prontos para uso."""
    documentos = carregar_documentos()
    return dividir_em_pedacos(documentos)


if __name__ == "__main__":
    documentos = carregar_documentos()
    pedacos = dividir_em_pedacos(documentos)
    print(f"Documentos carregados: {len(documentos)}")
    print(f"Pedaços gerados: {len(pedacos)}")
