from langchain_core.documents import Document

from src.tratamento.base_vetorial import criar_ou_carregar_vectorstore

_vectorstore = None


def _obter_vectorstore():
    """Recupera a vectorstore em memória, carregando uma única vez."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = criar_ou_carregar_vectorstore()
    return _vectorstore


def recuperar_contexto(pergunta: str, k: int = 5) -> list[tuple[Document, float]]:
    """Busca os k pedaços mais similares à pergunta no índice vetorial."""
    vectorstore = _obter_vectorstore()
    return vectorstore.similarity_search_with_score(pergunta, k=k)


def formatar_contexto(resultados: list[tuple[Document, float]]) -> str:
    """Monta o texto do contexto (conteúdo + fonte) para injetar no prompt."""
    blocos = []
    for i, (doc, score) in enumerate(resultados, start=1):
        fonte = doc.metadata.get("source", "desconhecida")
        linha = doc.metadata.get("row", "?")
        blocos.append(
            f"[{i}] (relevância: {score:.3f}, fonte: {fonte}, linha: {linha})\n{doc.page_content}"
        )
    return "\n\n".join(blocos)


def recuperar_contexto_formatado(pergunta: str, k: int = 5) -> str:
    """Atalho: recupera e formata o contexto em uma única chamada."""
    return formatar_contexto(recuperar_contexto(pergunta, k=k))


if __name__ == "__main__":
    import sys

    pergunta = " ".join(sys.argv[1:]) or "planeta em zona habitável"
    resultados = recuperar_contexto(pergunta)
    print(f"{len(resultados)} pedaços recuperados para: {pergunta!r}\n")
    print(formatar_contexto(resultados))
