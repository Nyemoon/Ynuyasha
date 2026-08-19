"""Testes herméticos da avaliação quantitativa.
"""

import pytest
from langchain_core.documents import Document

from src.tratamento import avaliacao

# ─── Fórmulas puras ──────────────────────────────────────────────────────────


def test_recall_completo_e_parcial():
    assert avaliacao.recall({"a", "b"}, ["a", "b", "c"]) == 1.0
    assert avaliacao.recall({"a", "b"}, ["a", "c"]) == pytest.approx(0.5)
    assert avaliacao.recall({"a"}, []) == 0.0


def test_recall_fora_da_base_sem_citacao():
    assert avaliacao.recall(set(), []) == 1.0
    assert avaliacao.recall(set(), ["a"]) == 0.0


def test_precisao_cita_correta():
    assert avaliacao.precisao({"a", "b"}, ["a", "x"]) == pytest.approx(0.5)
    assert avaliacao.precisao({"a"}, []) == 0.0
    assert avaliacao.precisao(set(), ["a"]) == 0.0
    assert avaliacao.precisao(set(), []) == 1.0


def test_mrr_primeiro_relevante():
    assert avaliacao.mrr({"b"}, ["a", "b", "c"]) == pytest.approx(0.5)
    assert avaliacao.mrr({"a"}, ["a", "b"]) == 1.0
    assert avaliacao.mrr({"z"}, ["a", "b"]) == 0.0


def test_ndcg_respeita_ordem_e_k():
    esperado = {"a", "b"}
    assert avaliacao.ndcg(esperado, ["a", "b"]) == pytest.approx(1.0)
    assert avaliacao.ndcg(esperado, ["x", "a", "b"]) < 1.0
    assert avaliacao.ndcg(esperado, ["a", "b", "c"], k=2) == pytest.approx(1.0)
    assert avaliacao.ndcg(set(), ["a"]) == 0.0


def test_hit_at_1():
    assert avaliacao.hit_at_1({"b"}, ["b", "a"]) == 1.0
    assert avaliacao.hit_at_1({"b"}, ["a", "b"]) == 0.0
    assert avaliacao.hit_at_1({"a"}, []) == 0.0


def test_metricas_caso_reune_todas():
    met = avaliacao.metricas_caso({"a", "b"}, ["a", "c"])
    assert set(met) == {"recall", "precisao", "mrr", "ndcg", "hit@1"}
    assert met["recall"] == pytest.approx(0.5)
    assert met["hit@1"] == 1.0


# ─── Parsing de citações ─────────────────────────────────────────────────────


def test_extrair_citacoes_varias_linhas():
    texto = (
        "Dados do planeta.\nFonte: planetas_e_estrelas_rag.csv, Linha 23\n"
        "Mais dados.\nFonte: planetas_e_estrelas_rag.csv, Linha 24"
    )
    assert avaliacao.extrair_citacoes(texto) == [
        ("planetas_e_estrelas_rag.csv", 23),
        ("planetas_e_estrelas_rag.csv", 24),
    ]


def test_extrair_citacoes_sem_texto():
    assert avaliacao.extrair_citacoes(None) == []
    assert avaliacao.extrair_citacoes("") == []


def test_citacoes_esperadas_com_e_sem_arquivo():
    caso = {"arquivo": "a.csv", "linhas_esperadas": [1, 2]}
    assert avaliacao.citacoes_esperadas(caso) == {("a.csv", 1), ("a.csv", 2)}
    assert avaliacao.citacoes_esperadas({"arquivo": None}) == set()


def test_citacoes_esperadas_aceita_arquivos_duplicados():
    caso = {
        "arquivos": ["a.csv", "b.csv"],
        "linhas_esperadas": [0],
    }
    assert avaliacao.citacoes_esperadas(caso) == {("a.csv", 0), ("b.csv", 0)}
    assert avaliacao.citacoes_esperadas({"arquivos": []}) == set()


# ─── Fora da base (recusa honesta via RAG) ───────────────────────────────────


def _docs_fake():
    return [
        (Document(page_content="parsec", metadata={"source": "glossario_astronomico_conceitos.csv", "row": 1}), 0.95),
        (Document(page_content="buraco", metadata={"source": "glossario_astronomico_conceitos.csv", "row": 7}), 0.80),
        (Document(page_content="m31", metadata={"source": "estrelas_e_objetos_simbad.csv", "row": 1}), 0.70),
    ]


def test_avaliar_fora_da_base_recusa_honesta():
    casos = [{"pergunta": "Qual a capital da França?"}]

    def recuperar_fake(pergunta, k=5):
        return []

    resultado = avaliacao.avaliar_fora_da_base(casos, recuperar=recuperar_fake)
    assert resultado["resumo"]["corretos"] == len(casos)
    assert all(d["recusa_honesta"] for d in resultado["casos"])
    assert all(d["citacoes"] == 0 for d in resultado["casos"])


def test_avaliar_fora_da_base_com_citacao_nao_e_recusa():
    casos = [{"pergunta": "lista de quasares"}]

    def recuperar_fake(pergunta, k=5):
        return [_docs_fake()[0]]

    resultado = avaliacao.avaliar_fora_da_base(casos, recuperar=recuperar_fake)
    assert resultado["resumo"]["corretos"] == 0
    assert resultado["casos"][0]["recusa_honesta"] is False


def test_avaliar_fora_da_base_erro_retorna_sem_quebrar():
    casos = [{"pergunta": "x"}]

    def recuperar_falha(pergunta, k=5):
        raise RuntimeError("Ollama indisponível")

    resultado = avaliacao.avaliar_fora_da_base(casos, recuperar=recuperar_falha)
    assert "erro" in resultado
    assert "Ollama" in resultado["erro"]


# ─── Retrieval (recuperar fake, sem Ollama) ──────────────────────────────────


def test_avaliar_retrieval_com_fake():
    caso = {"pergunta": "definição de parsec", "arquivo": "glossario_astronomico_conceitos.csv",
            "linhas_esperadas": [1]}

    def recuperar_fake(pergunta, k=3):
        return _docs_fake()[:k]

    resultado = avaliacao.avaliar_retrieval([caso], ks=(1, 3), recuperar=recuperar_fake)
    assert "erro" not in resultado
    resumo = resultado["resumo"]
    assert resumo["k=1"]["recall"] == pytest.approx(1.0)
    assert resumo["k=1"]["hit@1"] == 1.0
    assert resumo["k=3"]["recall"] == pytest.approx(1.0)


def test_avaliar_retrieval_erro_retorna_sem_quebrar():
    caso = {"pergunta": "x", "arquivo": "a.csv", "linhas_esperadas": [1]}

    def recuperar_falha(pergunta, k=3):
        raise RuntimeError("Ollama indisponível")

    resultado = avaliacao.avaliar_retrieval([caso], recuperar=recuperar_falha)
    assert "erro" in resultado
    assert "Ollama" in resultado["erro"]


def test_avaliar_retrieval_normaliza_source_absoluto():
    caso = {"pergunta": "definição de parsec", "arquivo": "glossario_astronomico_conceitos.csv",
            "linhas_esperadas": [1]}
    caminho_absoluto = "/maquina/qualquer/data/dataset/glossario_astronomico_conceitos.csv"

    def recuperar_absoluto(pergunta, k=3):
        return [(Document(page_content="parsec",
                          metadata={"source": caminho_absoluto, "row": 1}), 0.95)]

    resultado = avaliacao.avaliar_retrieval([caso], ks=(1, 3), recuperar=recuperar_absoluto)
    assert "erro" not in resultado
    assert resultado["resumo"]["k=1"]["recall"] == pytest.approx(1.0)
    assert resultado["resumo"]["k=1"]["hit@1"] == 1.0


# ─── Relatório Markdown ──────────────────────────────────────────────────────


def test_montar_relatorio_inclui_secoes_e_resumo():
    secoes = {
        "Retrieval (RAG)": {
            "resumo": {"k=1": {"recall": 1.0, "precisao": 1.0, "mrr": 1.0, "ndcg": 1.0, "hit@1": 1.0, "casos": 1, "corretos": 1}},
            "casos": [],
        },
    }
    metadados = {"data": "2026-08-13", "benchmark": "benchmark.json"}
    md = avaliacao.montar_relatorio(secoes, metadados)
    assert "# Relatório de Avaliação" in md
    assert "Retrieval (RAG)" in md
    assert "benchmark.json" in md


def test_salvar_relatorio_no_diretorio_tmp(tmp_path):
    secoes = {"Retrieval (RAG)": {
        "resumo": {"k=1": {"recall": 1.0, "casos": 0, "corretos": 0}},
        "casos": [],
    }}
    metadados = {"data": "2026-08-13", "benchmark": "b.json"}
    caminho = avaliacao.salvar_relatorio(secoes, metadados, diretorio=tmp_path)
    assert caminho.exists()
    assert caminho.read_text(encoding="utf-8").startswith("# Relatório de Avaliação")


# ─── registro de turnos e feedback ───────────────────────────────────────────


def test_registrar_turno_cria_e_anexa_csv(tmp_path):
    destino = tmp_path / "turnos.csv"
    avaliacao.registrar_turno("pergunta?", "resposta com Fonte: a.csv, Linha 1",
                              latencia_s=0.5, citou_fonte=True, modo="rag",
                              caminho=destino)
    avaliacao.registrar_turno("outra?", "resposta 2", latencia_s=0.1,
                              citou_fonte=False, modo="rag", caminho=destino)

    conteudo = destino.read_text(encoding="utf-8").strip().splitlines()
    assert conteudo[0].startswith("timestamp,modo,")
    assert len(conteudo) == 3
    assert line_contem(conteudo[1], "rag") and line_contem(conteudo[1], "0.500")


def _line_column(line, index):
    return line.split(",")[index]


def line_contem(line, sub):
    return sub in line


def test_registrar_feedback_escreve_linhas(tmp_path):
    destino = tmp_path / "feedback.csv"
    avaliacao.registrar_feedback("pergunta", "resposta", "positivo", caminho=destino)
    avaliacao.registrar_feedback("pergunta2", "resposta2", "negativo", caminho=destino)
    conteudo = destino.read_text(encoding="utf-8").strip().splitlines()
    assert len(conteudo) == 3
    assert "positivo" in conteudo[1]
    assert "negativo" in conteudo[2]


def test_registrar_turno_nao_quebra_em_erro(tmp_path, monkeypatch):
    def falhar(*args, **kwargs):
        raise OSError("disco cheio")

    monkeypatch.setattr("src.tratamento.avaliacao._append_csv", falhar)
    avaliacao.registrar_turno("p", "r", latencia_s=0.0, citou_fonte=False,
                              modo="rag", caminho=tmp_path / "x.csv")


# ─── agente._registrar_turno repassa metadados (todos em modo rag) ──────────


def test_agente_registrar_turno_default_rag(monkeypatch):
    from src.tratamento import agente

    capturados = {}

    def fake_registrar_turno(pergunta, resposta, **kwargs):
        capturados.update(kwargs)

    monkeypatch.setattr(agente, "LOGAR_TURNOS", True)
    monkeypatch.setattr("src.tratamento.avaliacao.registrar_turno", fake_registrar_turno)

    agente._registrar_turno("pergunta?", "resposta")
    assert capturados["modo"] == "rag"


def test_agente_responder_loga_turno_rag(monkeypatch):
    from src.tratamento import agente

    metadados_log = {}

    def fake_registrar_turno(pergunta, resposta, **kwargs):
        metadados_log.update(kwargs)

    monkeypatch.setattr(agente, "LOGAR_TURNOS", True)
    monkeypatch.setattr("src.tratamento.avaliacao.registrar_turno", fake_registrar_turno)
    monkeypatch.setattr(
        agente, "recuperar_contexto_com_apoio",
        lambda pergunta, k=5, historico=None: [],
    )
    monkeypatch.setattr(agente, "formatar_contexto", lambda resultados: "")
    monkeypatch.setattr(
        agente, "gerar_resposta",
        lambda pergunta, contexto, historico=None: "resposta com Fonte: a.csv, Linha 1",
    )

    agente.responder("pergunta?")
    assert metadados_log["modo"] == "rag"
    assert metadados_log["citou_fonte"] is True


# ─── citações no formato '## Fontes' (lista) ──────────────────────────────────


def test_extrair_citacoes_formato_markdown_fontes():
    texto = (
        "# Titulo\n\ncorpo da resposta\n\n## Fontes\n\n"
        "- ps-conf-ext-mapping.csv, linha 3\n"
        "- planetas_validados.txt, linha 8"
    )
    assert avaliacao.extrair_citacoes(texto) == [
        ("ps-conf-ext-mapping.csv", 3),
        ("planetas_validados.txt", 8),
    ]


def test_extrair_citacoes_mantem_formato_fonte_classico():
    texto = "Dados.\nFonte: habitabilidade_exoplanetas.csv, Linha 2\nMais."
    assert avaliacao.extrair_citacoes(texto) == [
        ("habitabilidade_exoplanetas.csv", 2),
    ]


# ─── avaliar_apoio (avaliação do corpus de data/documentos) ──────────────────


def test_avaliar_apoio_com_recuperar_fake():
    caso = {
        "pergunta": "Em que ano o Kepler-186 f foi descoberto?",
        "arquivo": "planetas_validados.txt",
        "linhas_esperadas": [8],
    }
    faux = Document(
        page_content="O planeta Kepler-186 f foi descoberto no ano de 2014.",
        metadata={"source": "planetas_validados.txt", "row": 8},
    )

    def recuperar_fake(pergunta, k=5):
        return [(faux, 12.0)]

    resultado = avaliacao.avaliar_apoio([caso], ks=(1, 3), recuperar=recuperar_fake)
    assert "erro" not in resultado
    assert resultado["resumo"]["k=1"]["recall"] == 1.0
    assert resultado["resumo"]["k=3"]["hit@1"] == 1.0
    assert resultado["casos"][0]["arquivo"] == "planetas_validados.txt"


def test_avaliar_apoio_caso_fora_da_base_exige_sem_citacoes():
    caso = {"pergunta": "Qual a capital da França?"}

    def recuperar_fake(pergunta, k=5):
        return []

    resultado = avaliacao.avaliar_apoio([caso], ks=(1,), recuperar=recuperar_fake)
    assert resultado["resumo"]["k=1"]["recall"] == 1.0
    assert resultado["casos"][0]["arquivo"] is None


def test_avaliar_apoio_default_mede_o_componente_direto(monkeypatch):
    caso = {
        "pergunta": "Em que ano o Kepler-186 f foi descoberto?",
        "arquivo": "planetas_validados.txt",
        "linhas_esperadas": [8],
    }
    faux = Document(
        page_content="O planeta Kepler-186 f foi descoberto no ano de 2014.",
        metadata={"source": "planetas_validados.txt", "row": 8},
    )
    chamadas = []

    class FalsoRecuperador:
        def buscar(self, pergunta, top_k=None, limiar=None):
            chamadas.append((pergunta, top_k, limiar))
            return [(faux, 12.0)]

    monkeypatch.setattr(
        "src.tratamento.documentos_apoio.obter_recuperador_apoio",
        lambda: FalsoRecuperador(),
    )
    resultado = avaliacao.avaliar_apoio([caso], ks=(1,))
    assert "erro" not in resultado
    assert resultado["resumo"]["k=1"]["recall"] == 1.0
    assert chamadas == [(caso["pergunta"], 1, 0.0)]
