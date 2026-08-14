"""Testes herméticos da avaliação quantitativa (Fase H).
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


# ─── Ferramentas (offline, um caso por dataset) ─────────────────────────────


def _caso(pergunta, ferramenta, args, arquivo, linhas):
    return {
        "pergunta": pergunta,
        "ferramenta": ferramenta,
        "args": args,
        "arquivo": arquivo,
        "linhas_esperadas": linhas,
    }


CASOS_FERRAMENTAS = [
    _caso("O que é um parsec?", "consultar_glossario", {"termo": "parsec"},
          "glossario_astronomico_conceitos.csv", [1]),
    _caso("Explique o que é um buraco negro.", "consultar_glossario",
          {"termo": "buraco negro"}, "glossario_astronomico_conceitos.csv", [7]),
    _caso("Quais são os planetas do sistema TRAPPIST-1?", "consultar_planetas",
          {"termo": "TRAPPIST-1"}, "planetas_e_estrelas_rag.csv", [23, 24, 25, 26, 27]),
    _caso("Quais planetas estão na zona habitável?", "consultar_habitabilidade",
          {"termo": "zona habitável"}, "habitabilidade_exoplanetas.csv", [2, 5, 10, 14, 15]),
    _caso("Dados do asteroide 433 Eros.", "consultar_asteroides", {"termo": "Eros"},
          "asteroides_cometas_jpl.csv", [0, 18]),
    _caso("Em que hemisfério se vê o Cruzeiro do Sul?", "consultar_constelacao",
          {"termo": "Cruzeiro do Sul"}, "constelacoes_iau.csv", [6]),
    _caso("Que tipo de objeto é M 31?", "consultar_objeto_simbad",
          {"termo": "M 31"}, "estrelas_e_objetos_simbad.csv", [1]),
    _caso("Qual a magnitude da estrela Gaia 5853498713190525696?",
          "consultar_estrelas_gaia", {"termo": "5853498713190525696"},
          "estrelas_proximas_gaia.csv", [0]),
    _caso("Quais eventos são supernovas?", "consultar_eventos", {"termo": "SN"},
          "eventos_transientes_extremos.csv", [28, 29, 30, 31, 32]),
]


def test_avaliar_ferramentas_offline_todas_corretas():
    resultado = avaliacao.avaliar_ferramentas(CASOS_FERRAMENTAS)
    resumo = resultado["resumo"]
    assert resumo["casos"] == len(CASOS_FERRAMENTAS)
    assert resumo["corretos"] == len(CASOS_FERRAMENTAS)
    assert resumo["recall"] == pytest.approx(1.0)
    assert resumo["precisao"] == pytest.approx(1.0)
    assert all(d["ok"] for d in resultado["casos"])


# ─── Fora da base (recusa honesta) ───────────────────────────────────────────


def test_avaliar_fora_da_base_recusa_honesta():
    casos = [
        {"pergunta": "Qual a capital da França?", "ferramenta": "consultar_glossario",
         "args": {"termo": "capital da França"}},
        {"pergunta": "Quem ganhou a última Copa do Mundo?", "ferramenta": "consultar_eventos",
         "args": {"termo": "futebol"}},
    ]
    resultado = avaliacao.avaliar_fora_da_base(casos)
    assert resultado["resumo"]["corretos"] == len(casos)
    assert all(d["recusa_honesta"] for d in resultado["casos"])
    assert all(d["citacoes"] == 0 for d in resultado["casos"])


# ─── Retrieval (recuperar fake, sem Ollama) ──────────────────────────────────


def _docs_fake():
    return [
        (Document(page_content="parsec", metadata={"source": "glossario_astronomico_conceitos.csv", "row": 1}), 0.95),
        (Document(page_content="buraco", metadata={"source": "glossario_astronomico_conceitos.csv", "row": 7}), 0.80),
        (Document(page_content="m31", metadata={"source": "estrelas_e_objetos_simbad.csv", "row": 1}), 0.70),
    ]


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
        "Ferramentas (offline)": avaliacao.avaliar_ferramentas(CASOS_FERRAMENTAS[:1]),
        "Retrieval (RAG)": {
            "resumo": {"k=1": {"recall": 1.0, "precisao": 1.0, "mrr": 1.0, "ndcg": 1.0, "hit@1": 1.0, "casos": 1, "corretos": 1}},
            "casos": [],
        },
    }
    metadados = {"data": "2026-08-13", "benchmark": "benchmark.json", "modo": "offline", "online": False}
    md = avaliacao.montar_relatorio(secoes, metadados)
    assert "# Relatório de Avaliação" in md
    assert "Ferramentas (offline)" in md
    assert "benchmark.json" in md


def test_salvar_relatorio_no_diretorio_tmp(tmp_path):
    secoes = {"Ferramentas (offline)": avaliacao.avaliar_ferramentas(CASOS_FERRAMENTAS[:1])}
    metadados = {"data": "2026-08-13", "benchmark": "b.json", "modo": "offline", "online": False}
    caminho = avaliacao.salvar_relatorio(secoes, metadados, diretorio=tmp_path)
    assert caminho.exists()
    assert caminho.read_text(encoding="utf-8").startswith("# Relatório de Avaliação")


# ─── registro de turnos e feedback (Fase I) ──────────────────────────────────


def test_registrar_turno_cria_e_anexa_csv(tmp_path):
    destino = tmp_path / "turnos.csv"
    avaliacao.registrar_turno("pergunta?", "resposta com Fonte: a.csv, Linha 1",
                              ferramentas_chamadas=2, latencia_s=0.5,
                              citou_fonte=True, modo="agente", caminho=destino)
    avaliacao.registrar_turno("outra?", "resposta 2", ferramentas_chamadas=0,
                              latencia_s=0.1, citou_fonte=False, modo="rag",
                              caminho=destino)

    conteudo = destino.read_text(encoding="utf-8").strip().splitlines()
    assert conteudo[0].startswith("timestamp,modo,")
    assert len(conteudo) == 3
    assert line_contem(conteudo[1], "agente") and line_contem(conteudo[1], "2")


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
    avaliacao.registrar_turno("p", "r", ferramentas_chamadas=0, latencia_s=0.0,
                              citou_fonte=False, modo="rag", caminho=tmp_path / "x.csv")


# ─── agente._registrar_turno repassa latência/metadados reais (Fase I) ──────


def test_agente_registrar_turno_usa_latencia_e_ferramentas_reais(monkeypatch, tmp_path):
    from src.tratamento import agente

    capturados = {}

    def fake_registrar_turno(pergunta, resposta, **kwargs):
        capturados.update(kwargs)

    monkeypatch.setattr(agente, "LOGAR_TURNOS", True)
    monkeypatch.setattr("src.tratamento.avaliacao.registrar_turno", fake_registrar_turno)

    agente._registrar_turno(
        "pergunta?",
        "resposta com Fonte: a.csv, Linha 1",
        metadados={"modo": "agente", "tool_calls": 3, "latencia_s": 1.25, "citou_fonte": True},
    )
    assert capturados["latencia_s"] == 1.25
    assert capturados["ferramentas_chamadas"] == 3
    assert capturados["modo"] == "agente"
    assert capturados["citou_fonte"] is True


def test_agente_registrar_turno_default_rag_sem_metadados(monkeypatch):
    from src.tratamento import agente

    capturados = {}

    def fake_registrar_turno(pergunta, resposta, **kwargs):
        capturados.update(kwargs)

    monkeypatch.setattr(agente, "LOGAR_TURNOS", True)
    monkeypatch.setattr("src.tratamento.avaliacao.registrar_turno", fake_registrar_turno)

    agente._registrar_turno("pergunta?", "resposta")
    assert capturados["modo"] == "rag"
    assert capturados["ferramentas_chamadas"] == 0


def test_agente_responder_loga_metadados_do_agente(monkeypatch):
    from src.tratamento import agente

    metadados_log = {}

    def fake_registrar_turno(pergunta, resposta, **kwargs):
        metadados_log.update(kwargs)

    monkeypatch.setattr(agente, "LOGAR_TURNOS", True)
    monkeypatch.setattr("src.tratamento.avaliacao.registrar_turno", fake_registrar_turno)

    def fake_executar(pergunta, historico=None, thread_id=None, registrador=None):
        registrador({"modo": "agente", "tool_calls": 2, "latencia_s": 0.75, "citou_fonte": True})
        return "resposta final"

    monkeypatch.setattr("src.tratamento.agente_ia.executar_agente", fake_executar)
    monkeypatch.setattr("src.tratamento.agente_ia.groq_disponivel", lambda: True)

    agente.responder("pergunta?")
    assert metadados_log["latencia_s"] == 0.75
    assert metadados_log["ferramentas_chamadas"] == 2
    assert metadados_log["citou_fonte"] is True
