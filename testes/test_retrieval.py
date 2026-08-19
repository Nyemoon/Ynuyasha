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
    recuperar_contexto_com_apoio,
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
    # Documento que só aparece no BM25 não tem vetor gravado na vectorstore:
    # sem escore cosseno fica abaixo do limiar e é descartado (sem re-embedar).
    assert [doc for doc, _ in resultado] == [a]
    assert embeddings.chamadas["query"] == 0
    assert embeddings.chamadas["documents"] == 0


def test_recuperar_contexto_bm25_usa_escore_da_vectorstore(limiar, monkeypatch):
    a = _doc("na vectorstore e no BM25", linha=1)
    vs = FakeVectorStore([(a, 0.9)])
    bm25 = FakeBM25([a])
    embeddings = FakeEmbeddings()
    monkeypatch.setattr(retrieval, "embeddings_model", embeddings)

    resultado = recuperar_contexto("pergunta", k=5, vectorstore=vs, bm25=bm25)
    # Todo documento da base está na vectorstore; o candidato do BM25 recebe o
    # escore cosseno já gravado — nada é re-embebido.
    assert [doc for doc, _ in resultado] == [a]
    assert [score for _, score in resultado] == pytest.approx([0.9])
    assert embeddings.chamadas["query"] == 0
    assert embeddings.chamadas["documents"] == 0


# ─── k dinâmico para enumeração (k_para_pergunta) ────────────────────────────


def test_k_para_pergunta_eleva_k_em_enumercao():
    assert retrieval.k_para_pergunta("Liste planetas descobertos por trânsito", 5) == retrieval.K_ENUMERACAO
    assert retrieval.k_para_pergunta("Quais planetas estão na zona habitável?", 5) == retrieval.K_ENUMERACAO
    assert retrieval.k_para_pergunta("Quantas estrelas tem a Lira?", 5) == retrieval.K_ENUMERACAO
    assert retrieval.k_para_pergunta("Todos os quasares SDSS", 5) == retrieval.K_ENUMERACAO


def test_k_para_pergunta_mantem_k_em_fato_unico():
    assert retrieval.k_para_pergunta("Qual a paralaxe da estrela Proxima Centauri?", 5) == 5
    assert retrieval.k_para_pergunta("O que é um parsec?", 5) == 5
    assert retrieval.k_para_pergunta("Qual o tipo de objeto M 31?", 3) == 3
    assert retrieval.k_para_pergunta("", 5) == 5
    assert retrieval.k_para_pergunta(None, 5) == 5


def test_recuperar_contexto_respeita_k(limiar):
    docs = [_doc(f"doc {i}", linha=i) for i in range(4)]
    pares = [(doc, 0.9 - i * 0.1) for i, doc in enumerate(docs)]
    vs = FakeVectorStore(pares)
    bm25 = FakeBM25([])

    resultado = recuperar_contexto("pergunta", k=2, vectorstore=vs, bm25=bm25)
    assert len(resultado) == 2
    assert [doc for doc, _ in resultado] == [docs[0], docs[1]]


# ─── subconjunto-por-atributo (caminho determinístico) ───────────────────────
# Estes testes leem os CSVs reais em data/dataset (não usam embeddings).


def test_exatos_por_atributo_zona_habitavel():
    resultado = retrieval._recuperar_exatos_por_atributo(
        "Quais planetas estão na zona habitável?"
    )
    assert resultado is not None
    linhas = [doc.metadata["row"] for doc, _ in resultado]
    assert linhas == [2, 5, 10, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28]
    assert all(score == 1.0 for _, score in resultado)


def test_exatos_por_atributo_asteroides_perigosos():
    resultado = retrieval._recuperar_exatos_por_atributo(
        "Quais asteroides são potencialmente perigosos?"
    )
    linhas = [doc.metadata["row"] for doc, _ in resultado]
    assert linhas == [5, 7, 10]


def test_exatos_por_atributo_eventos_supernovas():
    resultado = retrieval._recuperar_exatos_por_atributo(
        "Quais eventos são supernovas?"
    )
    linhas = [doc.metadata["row"] for doc, _ in resultado]
    assert linhas == list(range(28, 42))


def test_exatos_por_atributo_transito():
    resultado = retrieval._recuperar_exatos_por_atributo(
        "Liste planetas descobertos pelo método de trânsito."
    )
    linhas = [doc.metadata["row"] for doc, _ in resultado]
    assert linhas == [4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30]


def test_exatos_por_atributo_fato_entidade_zona():
    resultado = retrieval._recuperar_exatos_por_atributo(
        "O TRAPPIST-1 e está na zona habitável?"
    )
    assert resultado is not None
    assert [(doc.metadata["source"], doc.metadata["row"]) for doc, _ in resultado] == [
        ("habitabilidade_exoplanetas.csv", 26)
    ]


def test_exatos_por_atributo_fato_entidade_fora_da_zona():
    resultado = retrieval._recuperar_exatos_por_atributo(
        "O TRAPPIST-1 b está na zona habitável?"
    )
    assert [(doc.metadata["source"], doc.metadata["row"]) for doc, _ in resultado] == [
        ("habitabilidade_exoplanetas.csv", 23)
    ]


def test_exatos_por_atributo_fato_entidade_transito():
    resultado = retrieval._recuperar_exatos_por_atributo(
        "O Kepler-452 b foi descoberto por trânsito?"
    )
    assert [(doc.metadata["source"], doc.metadata["row"]) for doc, _ in resultado] == [
        ("planetas_e_estrelas_rag.csv", 19)
    ]


def test_exatos_por_atributo_fato_entidade_asteroide():
    resultado = retrieval._recuperar_exatos_por_atributo(
        "O 433 Eros é potencialmente perigoso?"
    )
    assert [(doc.metadata["source"], doc.metadata["row"]) for doc, _ in resultado] == [
        ("asteroides_cometas_jpl.csv", 0)
    ]


def test_exatos_por_atributo_fato_sem_entidade_volta_ao_ranking():
    sem_entidade = retrieval._recuperar_exatos_por_atributo(
        "O planeta mais próximo da zona habitável?"
    )
    assert sem_entidade is None


def test_exatos_por_atributo_nao_dispara_sem_entidade_nem_padrao():
    sem_entidade = retrieval._recuperar_exatos_por_atributo(
        "explica o método de trânsito"
    )
    sem_padrao = retrieval._recuperar_exatos_por_atributo(
        "Qual a paralaxe da estrela Proxima Centauri?"
    )
    assert sem_entidade is None
    assert sem_padrao is None


class FakeVectorStoreProibido:
    def similarity_search_with_score(self, pergunta, k=10):
        raise AssertionError("vectorstore não deveria ser consultada")


def test_recuperar_contexto_atributo_ignora_vectorstore_e_bm25(limiar):
    resultado = recuperar_contexto(
        "Quais asteroides são potencialmente perigosos?",
        k=5,
        vectorstore=FakeVectorStoreProibido(),
        bm25=FakeBM25([]),
    )
    linhas = [doc.metadata["row"] for doc, _ in resultado]
    assert linhas == [5, 7, 10]


# ─── recusa honesta: trecho fraco sem sobreposição lexical ───────────────────
# Perguntas fora da base às vezes "encaixam" num trecho com escore fraco
# (0.65–0.68). O gate exige que a pergunta compartilhe ao menos um token de
# conteúdo com o trecho; sem nenhum, o resultado vira vazio (recusa na geração).


@pytest.fixture
def limiar_forca(monkeypatch):
    monkeypatch.setattr(retrieval, "LIMIAR_FORCA", 0.68)
    return 0.68


def test_recusar_fraco_sem_sobreposicao_detem_ruido(limiar_forca):
    doc = _doc(
        "A constelação de Ursa Maior é visível no hemisfério Sul.",
        fonte="constelacoes_iau.csv",
        linha=5,
    )
    resultado = [(doc, 0.658)]
    assert (
        retrieval._recusar_fraco_sem_sobreposicao(
            "Quem ganhou a última Copa do Mundo?", resultado
        )
        is True
    )


def test_recusar_fraco_sem_sobreposicao_aceita_forte(limiar_forca):
    doc = _doc(
        "A constelação de Ursa Maior é visível no hemisfério Sul.",
        fonte="constelacoes_iau.csv",
        linha=5,
    )
    resultado = [(doc, 0.90)]
    assert (
        retrieval._recusar_fraco_sem_sobreposicao(
            "Quem ganhou a última Copa do Mundo?", resultado
        )
        is False
    )


def test_recusar_fraco_sem_sobreposicao_aceita_fraco_com_tokens(limiar_forca):
    doc = _doc(
        "O objeto astronômico M 31 é classificado como galáxia.",
        fonte="estrelas_e_objetos_simbad.csv",
        linha=1,
    )
    resultado = [(doc, 0.66)]
    assert (
        retrieval._recusar_fraco_sem_sobreposicao(
            "Que tipo de objeto é M 31?", resultado
        )
        is False
    )


def test_recuperar_contexto_recusa_ruido_sem_sobreposicao(limiar, limiar_forca):
    doc = _doc(
        "A constelação de Ursa Maior é visível no hemisfério Sul.",
        fonte="constelacoes_iau.csv",
        linha=5,
    )
    vs = FakeVectorStore([(doc, 0.658)])
    bm25 = FakeBM25([])
    resultado = recuperar_contexto(
        "Quem ganhou a última Copa do Mundo?", vectorstore=vs, bm25=bm25
    )
    assert resultado == []


def test_recuperar_contexto_mantem_fraco_com_sobreposicao(limiar, limiar_forca):
    doc = _doc(
        "O objeto astronômico M 31 é classificado como galáxia.",
        fonte="estrelas_e_objetos_simbad.csv",
        linha=1,
    )
    vs = FakeVectorStore([(doc, 0.66)])
    bm25 = FakeBM25([])
    resultado = recuperar_contexto(
        "Que tipo de objeto é M 31?", vectorstore=vs, bm25=bm25
    )
    assert resultado == [(doc, 0.66)]


def test_recuperar_contexto_aceita_forte_sem_sobreposicao(limiar, limiar_forca):
    doc = _doc(
        "A constelação de Ursa Maior é visível no hemisfério Sul.",
        fonte="constelacoes_iau.csv",
        linha=5,
    )
    vs = FakeVectorStore([(doc, 0.90)])
    bm25 = FakeBM25([])
    resultado = recuperar_contexto(
        "Quem ganhou a última Copa do Mundo?", vectorstore=vs, bm25=bm25
    )
    assert resultado == [(doc, 0.90)]


# ─── merge com o corpus de apoio (recuperar_contexto_com_apoio) ──────────────


class FakeRecuperadorApoio:
    def __init__(self, pares):
        self._pares = pares

    def buscar(self, pergunta, top_k=10, limiar=0.0):
        return [(doc, score) for doc, score in self._pares][:top_k]


def test_contexto_com_apoio_anexa_apoio_depois_da_principal(limiar):
    principal = _doc("planeta na base principal", linha=1)
    apoio_doc = _doc(
        "coluna disc_year do catálogo",
        fonte="ps-conf-ext-mapping.csv",
        linha=14,
    )
    vs = FakeVectorStore([(principal, 0.9)])
    bm25 = FakeBM25([principal])
    rec_apoio = FakeRecuperadorApoio([(apoio_doc, 15.0)])

    resultado = recuperar_contexto_com_apoio(
        "pergunta", vectorstore=vs, bm25=bm25, recuperador_apoio=rec_apoio
    )
    assert [doc for doc, _ in resultado] == [principal, apoio_doc]


def test_contexto_com_apoio_nao_duplica_mesma_chave(limiar):
    principal = _doc("planeta na principal", linha=1)
    duplicado = _doc("mesma chave (source,row) vinda do apoio", linha=1)
    vs = FakeVectorStore([(principal, 0.9)])
    bm25 = FakeBM25([principal])
    rec_apoio = FakeRecuperadorApoio([(duplicado, 15.0)])

    resultado = recuperar_contexto_com_apoio(
        "pergunta", vectorstore=vs, bm25=bm25, recuperador_apoio=rec_apoio
    )
    assert len(resultado) == 1
    assert resultado[0][0] is principal


def test_contexto_com_apoio_respeita_teto_max_apoio(limiar, monkeypatch):
    from src.tratamento import retrieval

    monkeypatch.setattr(retrieval, "MAX_APOIO_CONTEXTO", 2)
    principal = _doc("principal", linha=1)
    apoio_1 = _doc("apoio um", fonte="m.csv", linha=10)
    apoio_2 = _doc("apoio dois", fonte="m.csv", linha=20)
    apoio_3 = _doc("apoio tres", fonte="m.csv", linha=30)
    vs = FakeVectorStore([(principal, 0.9)])
    bm25 = FakeBM25([principal])
    rec_apoio = FakeRecuperadorApoio([(apoio_1, 10.0), (apoio_2, 9.0), (apoio_3, 8.0)])

    resultado = recuperar_contexto_com_apoio(
        "pergunta", vectorstore=vs, bm25=bm25, recuperador_apoio=rec_apoio
    )
    assert [doc for doc, _ in resultado] == [principal, apoio_1, apoio_2]


# ─── Retrieval conversacional (_query_com_contexto) ──────────────────────────


def test_query_com_contexto_sem_historico_mantem_pergunta():
    assert retrieval._query_com_contexto("O que é um parsec?") == "O que é um parsec?"
    assert retrieval._query_com_contexto("O que é um parsec?", None) == "O que é um parsec?"
    assert retrieval._query_com_contexto("O que é um parsec?", []) == "O que é um parsec?"


def test_query_com_contexto_referencia_curta_expande():
    historico = [
        {"role": "user", "content": "O que é um parsec?"},
        {"role": "assistant", "content": "Parsec é uma unidade de distância que vale 3,26 anos-luz."},
    ]
    expandida = retrieval._query_com_contexto("e quanto vale em anos-luz?", historico)
    assert expandida.startswith("e quanto vale em anos-luz? | ")
    assert "parsec" in expandida
    assert "3,26 anos-luz" in expandida


def test_query_com_contexto_deitico_expande():
    historico = [
        {"role": "user", "content": "O TRAPPIST-1 e está na zona habitável?"},
        {"role": "assistant", "content": "Sim, está entre as candidatas."},
    ]
    expandida = retrieval._query_com_contexto("esse planeta tem água?", historico)
    assert "TRAPPIST-1 e" in expandida


def test_query_com_contexto_pivo_claro_nao_expande():
    historico = [
        {"role": "user", "content": "O que é um parsec?"},
        {"role": "assistant", "content": "Parsec é uma unidade de distância em astronomia."},
    ]
    assert (
        retrieval._query_com_contexto(
            "O que é a matéria escura?", historico
        )
        == "O que é a matéria escura?"
    )


def test_query_com_contexto_ignora_acentos_na_deteccao():
    historico = [
        {"role": "user", "content": "Qual a temperatura do TRAPPIST-1 e?"},
        {"role": "assistant", "content": "Cerca de 249,7 K."},
    ]
    expandida = retrieval._query_com_contexto("e sobre a composição dele?", historico)
    assert "TRAPPIST-1 e" in expandida


def test_query_com_contexto_usa_ultimas_trocas_com_limites():
    historico = [
        {"role": "user", "content": "pergunta antiga A"},
        {"role": "assistant", "content": "resposta antiga A"},
        {"role": "user", "content": "pergunta antiga B"},
        {"role": "assistant", "content": "resposta antiga B"},
        {"role": "user", "content": "pergunta da vez"},
        {"role": "assistant", "content": "resposta da vez"},
    ]
    expandida = retrieval._query_com_contexto("e daí?", historico)
    assert "pergunta antiga A" not in expandida
    assert "pergunta antiga B" in expandida
    assert "pergunta da vez" in expandida
    assert len(expandida) <= len("e daí? | ") + retrieval.MAX_TAMANHO_CONTEXTO_BUSCA


def test_query_com_contexto_lida_com_blocos_do_gradio():
    historico = [
        {"role": "user", "content": [{"type": "text", "text": "O que é um parsec?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Parsec é uma unidade."}]},
    ]
    expandida = retrieval._query_com_contexto("e vale quanto?", historico)
    assert "O que é um parsec?" in expandida
    assert "Parsec é uma unidade." in expandida
