import pytest
from langchain_core.documents import Document

from src.tratamento import retrieval
from src.tratamento.retrieval import (
    LIMIAR_RELEVANCIA,
    _chave_documento,
    _cosseno,
    _fusao_rrf,
    filtrar_por_relevancia,
    formatar_contexto,
    recuperar_contexto,
)


def _doc(texto, fonte="fake.csv", linha=0):
    return Document(page_content=texto, metadata={"source": fonte, "row": linha})


class FakeVectorStore:
    def __init__(self, pares):
        self._pares = pares

    def similarity_search_with_score(self, pergunta, k=10):
        return self._pares


class FakeBM25:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, pergunta):
        return self._docs


class FakeEmbeddings:
    def __init__(self):
        self.chamadas = {"query": 0, "documents": 0}

    def embed_query(self, pergunta):
        self.chamadas["query"] += 1
        return [1.0, 0.0]

    def embed_documents(self, textos):
        self.chamadas["documents"] += 1
        return [[1.0, 0.0] for _ in textos]


class FakeEmbeddingsProibido:
    def embed_query(self, pergunta):
        raise AssertionError("embeddings não deveriam ser chamados")

    def embed_documents(self, textos):
        raise AssertionError("embeddings não deveriam ser chamados")


@pytest.fixture
def limiar(monkeypatch):
    monkeypatch.setattr(retrieval, "LIMIAR_RELEVANCIA", 0.5)
    return 0.5


@pytest.fixture(autouse=True)
def sem_embeddings_reais(monkeypatch):
    monkeypatch.setattr(retrieval, "embeddings_model", FakeEmbeddingsProibido())


def test_limiar_default_importado():
    assert isinstance(LIMIAR_RELEVANCIA, float)


def test_filtrar_por_relevancia_mantem_score_igual_ao_limiar():
    doc = _doc("texto")
    resultado = filtrar_por_relevancia([(doc, 0.5)], 0.5)
    assert resultado == [(doc, 0.5)]


def test_filtrar_por_relevancia_descarta_abaixo():
    mantido = _doc("mantido", linha=1)
    descartado = _doc("descartado", linha=2)
    resultado = filtrar_por_relevancia([(mantido, 0.9), (descartado, 0.4)], 0.5)
    assert resultado == [(mantido, 0.9)]


def test_fusao_rrf_prioriza_documento_bem_posicionado():
    a, b, c = _doc("a", linha=1), _doc("b", linha=2), _doc("c", linha=3)
    fundidos = _fusao_rrf([[a, b, c], [b, c, a]])
    assert [d.page_content for d in fundidos] == ["b", "a", "c"]


def test_fusao_rrf_nao_duplica_documento():
    a, b, c = _doc("a", linha=1), _doc("b", linha=2), _doc("c", linha=3)
    fundidos = _fusao_rrf([[a, b], [a, c]])
    assert len(fundidos) == 3
    assert fundidos.count(a) == 1


def test_chave_documento_usa_source_e_row():
    doc = _doc("texto", fonte="planetas.csv", linha=7)
    assert _chave_documento(doc) == ("planetas.csv", 7)


def test_chave_documento_fallback_para_hash():
    doc = Document(page_content="sem metadados")
    assert _chave_documento(doc) == ("conteudo", hash("sem metadados"))


def test_cosseno_identico_e_ortogonal():
    assert _cosseno([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosseno([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_formatar_contexto_inclui_metadados():
    doc = _doc("buraco negro", fonte="glossario.csv", linha=3)
    texto = formatar_contexto([(doc, 0.87)])
    assert "glossario.csv" in texto
    assert "linha: 3" in texto
    assert "0.870" in texto
    assert "buraco negro" in texto


def test_formatar_contexto_vazio():
    assert formatar_contexto([]) == ""


def test_recuperar_contexto_com_fakes_sem_embeddings(limiar):
    a = _doc("planeta em zona habitável", linha=1)
    b = _doc("exoplaneta quente", linha=2)
    vs = FakeVectorStore([(a, 0.9), (b, 0.8)])
    bm25 = FakeBM25([a, b])

    resultado = recuperar_contexto(
        "planeta em zona habitável", k=5, vectorstore=vs, bm25=bm25
    )
    assert [doc for doc, _ in resultado] == [a, b]
    assert [score for _, score in resultado] == pytest.approx([0.9, 0.8])


def test_recuperar_contexto_filtra_abaixo_limiar(limiar):
    a = _doc("fora da base", linha=1)
    vs = FakeVectorStore([(a, 0.3)])
    bm25 = FakeBM25([a])

    resultado = recuperar_contexto("fora da base", vectorstore=vs, bm25=bm25)
    assert resultado == []


def test_recuperar_contexto_bm25_somente_usa_embeddings(limiar, monkeypatch):
    a = _doc("planeta na base", linha=1)
    b = _doc("achado só no BM25", linha=2)
    vs = FakeVectorStore([(a, 0.9)])
    bm25 = FakeBM25([b])
    embeddings = FakeEmbeddings()
    monkeypatch.setattr(retrieval, "embeddings_model", embeddings)

    resultado = recuperar_contexto("pergunta", k=5, vectorstore=vs, bm25=bm25)
    assert [doc for doc, _ in resultado] == [a, b]
    assert [score for _, score in resultado] == pytest.approx([0.9, 1.0])
    assert embeddings.chamadas["query"] == 1
    assert embeddings.chamadas["documents"] == 1


def test_recuperar_contexto_respeita_k(limiar):
    docs = [_doc(f"doc {i}", linha=i) for i in range(4)]
    pares = [(doc, 0.9 - i * 0.1) for i, doc in enumerate(docs)]
    vs = FakeVectorStore(pares)
    bm25 = FakeBM25([])

    resultado = recuperar_contexto("pergunta", k=2, vectorstore=vs, bm25=bm25)
    assert len(resultado) == 2
    assert [doc for doc, _ in resultado] == [docs[0], docs[1]]
