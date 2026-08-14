import os

import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores.utils import _cosine_similarity
from rank_bm25 import BM25Okapi

from src.tratamento.base_vetorial import criar_ou_carregar_vectorstore
from src.tratamento.embeddings import embeddings_model
from src.tratamento.loading import obter_pedacos

_vectorstore = None
_bm25 = None

# Cache do vetor da pergunta: o Gradio chama preparar_contexto para exibir o
# contexto e depois reutiliza o resultado na geração; sem este cache a pergunta
# seria re-embebida (e os candidatos só-BM25 re-escoreados) desnecessariamente.
_ultima_query = None
_ultimo_vetor = None
_ultimo_k = None


def _embedar_pergunta(pergunta: str, k: int) -> list:
    """Embede a pergunta com cache: evita re-embedar a mesma query no mesmo k."""
    global _ultima_query, _ultimo_vetor, _ultimo_k
    if pergunta == _ultima_query and k == _ultimo_k and _ultimo_vetor is not None:
        return _ultimo_vetor
    _ultima_query = pergunta
    _ultimo_k = k
    _ultimo_vetor = embeddings_model.embed_query(pergunta)
    return _ultimo_vetor

LIMIAR_RELEVANCIA = float(os.getenv("RAG_LIMIAR_RELEVANCIA", "0.65"))
TOP_CANDIDATOS = 10  # quantos resultados cada busca traz antes da fusão RRF
CONSTANTE_RRF = 60  # constante padrão do Reciprocal Rank Fusion
# Limite de candidatos só-BM25 re-escoreados por embedding no Ollama de CPU.
# Cada documento custa ~16s no hardware atual; o sem_score já vem ordenado pela
# fusão RRF, então re-embedar apenas o topo basta para preencher os k finais.
MAX_REESCORE = 2


def _obter_vectorstore():
    """Recupera a vectorstore em memória, carregando uma única vez."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = criar_ou_carregar_vectorstore()
    return _vectorstore


class BM25Recuperador:
    """Retriever léxico BM25 local, substituto do BM25Retriever do
    langchain-community (que está em sunset).

    Reproduz o comportamento original: tokenização por quebra de espaços,
    BM25Okapi e `.invoke()` devolvendo os `k` documentos mais relevantes.
    """

    def __init__(self, documentos, vectorizer):
        self.docs = list(documentos)
        self.vectorizer = vectorizer
        self.k = TOP_CANDIDATOS

    @classmethod
    def from_documents(cls, documentos):
        corpus = [doc.page_content.split() for doc in documentos]
        return cls(documentos, BM25Okapi(corpus))

    def invoke(self, pergunta):
        return self.vectorizer.get_top_n(pergunta.split(), self.docs, n=self.k)


def _obter_bm25() -> BM25Recuperador:
    """Recupera o retriever lexical BM25, construído uma única vez.

    Usa os mesmos pedaços (com templates semânticos) da Fase de ingestão.
    """
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Recuperador.from_documents(obter_pedacos())
    return _bm25


def _chave_documento(doc: Document) -> tuple:
    """Identidade estável de um Documento para a fusão entre recuperadores.

    Prioriza (source, row) — a mesma convenção usada nas citações — e recorre
    ao hash do conteúdo quando os metadados não estiverem disponíveis.
    """
    fonte = doc.metadata.get("source")
    linha = doc.metadata.get("row")
    if fonte and linha is not None:
        return (str(fonte), int(linha))
    return ("conteudo", hash(doc.page_content))


def _cosseno(vetor_a: list[float], vetor_b: list[float]) -> float:
    """Similaridade cosseno entre dois vetores.

    Reutiliza a mesma função da vectorstore (InMemoryVectorStore), garantindo
    que o score manual esteja na mesma escala do limiar de relevância.
    """
    matriz = _cosine_similarity(np.array([vetor_a]), np.array([vetor_b]))
    return float(matriz[0][0])


def filtrar_por_relevancia(
    resultados: list[tuple[Document, float]], limiar: float
) -> list[tuple[Document, float]]:
    """Mantém apenas (doc, score) com score >= limiar."""
    return [(doc, score) for doc, score in resultados if score >= limiar]


def _fusao_rrf(
    listas: list[list[Document]], constante: int = CONSTANTE_RRF
) -> list[Document]:
    """Funde listas ordenadas de Documentos via Reciprocal Rank Fusion (RRF).

    Documentos que aparecem bem posicionados em várias listas sobem no ranking
    final. A identidade de cada Documento é dada por _chave_documento.
    """
    soma = {}
    docs_por_chave = {}
    for lista in listas:
        for posicao, doc in enumerate(lista):
            chave = _chave_documento(doc)
            soma[chave] = soma.get(chave, 0.0) + 1.0 / (constante + posicao + 1)
            docs_por_chave.setdefault(chave, doc)
    ordenados = sorted(soma.items(), key=lambda item: item[1], reverse=True)
    return [docs_por_chave[chave] for chave, _ in ordenados]


def recuperar_contexto(
    pergunta: str,
    k: int = 5,
    vectorstore=None,
    bm25: BM25Recuperador | None = None,
) -> list[tuple[Document, float]]:
    """Busca os trechos mais relevantes combinando busca semântica e lexical.

    1. Recupera top-TOP_CANDIDATOS da vectorstore (com score) e do BM25.
    2. Funde os dois rankings por RRF.
    3. Atribui score cosseno aos candidatos só-BM25 mais bem posicionados na
       fusão (até MAX_REESCORE documentos, um lote de embeddings), mantendo o
       limiar de relevância aplicável a todos.
    4. Filtra por LIMIAR_RELEVANCIA e devolve os k melhores.

    Se nenhum trecho atingir o limiar, retorna lista vazia — indicando que a
    pergunta não consta na base de conhecimento.
    """
    vs = vectorstore or _obter_vectorstore()
    bm = bm25 or _obter_bm25()

    # Quando o store aceita busca por vetor pré-computado, embute a pergunta uma
    # única vez e reaproveita o vetor também no re-escore de candidatos só do BM25
    # (evita o segundo embedding da pergunta no caminho mais lento).
    usa_vetor_unico = hasattr(vs, "similarity_search_with_score_by_vector")
    vetor_pergunta = None
    if usa_vetor_unico:
        vetor_pergunta = _embedar_pergunta(pergunta, k)
        candidatos_vetorial = vs.similarity_search_with_score_by_vector(
            vetor_pergunta, k=TOP_CANDIDATOS
        )
    else:
        candidatos_vetorial = vs.similarity_search_with_score(pergunta, k=TOP_CANDIDATOS)
    candidatos_bm25 = bm.invoke(pergunta)

    fundidos = _fusao_rrf(
        [[doc for doc, _ in candidatos_vetorial], candidatos_bm25]
    )

    score_por_chave = {
        _chave_documento(doc): score for doc, score in candidatos_vetorial
    }

    sem_score = [
        doc for doc in fundidos if _chave_documento(doc) not in score_por_chave
    ]
    if sem_score:
        if vetor_pergunta is None:
            vetor_pergunta = _embedar_pergunta(pergunta, k)
        # Re-escoreia só o topo da fusão: embeder todos os candidatos só-BM25
        # custaria ~16s cada no Ollama de CPU (vetores longos). Os excedentes
        # ficam com score 0 (abaixo do limiar) — nunca entram no resultado.
        alvo = sem_score[:MAX_REESCORE]
        vetores = embeddings_model.embed_documents([doc.page_content for doc in alvo])
        for doc, vetor in zip(alvo, vetores):
            score_por_chave[_chave_documento(doc)] = _cosseno(vetor_pergunta, vetor)
        for doc in sem_score[MAX_REESCORE:]:
            score_por_chave[_chave_documento(doc)] = 0.0

    resultados = [(doc, score_por_chave[_chave_documento(doc)]) for doc in fundidos]
    return filtrar_por_relevancia(resultados, LIMIAR_RELEVANCIA)[:k]


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
    print(
        f"{len(resultados)} pedaços recuperados para: {pergunta!r} "
        f"(limiar: {LIMIAR_RELEVANCIA})\n"
    )
    texto = formatar_contexto(resultados)
    if texto:
        print(texto)
    else:
        print("Nenhum trecho relevante encontrado na base de conhecimento.")
