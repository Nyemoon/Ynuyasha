import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.tratamento import documentos_apoio as apoio

RAIZ = Path(__file__).resolve().parents[1]

# Marcas que cada linha esperada do benchmark 'apoio' deve conter (guard test).
_MARCAS_LINHAS = {
    ("planetas_validados.txt", 8): "Kepler-186",
    ("planetas_validados.txt", 3): "Proxima Cen b",
    ("planetas_validados.txt", 18): "K2-18 b",
    ("planetas_validados.txt", 4): "TRAPPIST-1 b",
    ("planetas_validados.txt", 27): "TRAPPIST-1 e",
    ("ps-conf-ext-mapping.csv", 0): "pl_name",
    ("ps-conf-ext-mapping.csv", 14): "disc_year",
    ("ps-conf-ext-mapping.csv", 54): "pl_massj",
    ("Exoplanet_Archive_Column_Mapping_CSV.csv", 0): "fpl_name",
    ("Exoplanet_Archive_Column_Mapping_CSV.csv", 14): "pl_disc",
    ("old-comp-new-comp-mapping.csv", 0): "fpl_name",
    ("conf-comp-ext-not-in-ps.csv", 0): "pl_kepflag",
}


def _doc(texto, fonte, linha):
    return Document(
        page_content=texto, metadata={"source": fonte, "row": linha}
    )


# ─── parsers (herméticos, com arquivos locais) ───────────────────────────────


def test_parse_planetas_validados_ignora_cabecalho_e_separadores(tmp_path):
    arquivo = tmp_path / "planetas_validados.txt"
    arquivo.write_text(
        "===============================================================\n"
        "Nome do Planeta      | Estrela Hospedeira   | Ano de Descoberta\n"
        "===============================================================\n"
        "51 Peg b             | 51 Peg               | 1995\n"
        "55 Cnc e             | 55 Cnc               | 2004\n",
        encoding="utf-8",
    )
    docs = apoio._parse_planetas_validados(arquivo)
    assert len(docs) == 2
    assert docs[0].metadata == {"source": "planetas_validados.txt", "row": 0}
    assert "51 Peg" in docs[0].page_content
    assert "1995" in docs[0].page_content
    assert docs[1].metadata["row"] == 1


def test_parse_mapeamento_separa_tabelas_e_ignora_filler(tmp_path):
    arquivo = tmp_path / "mapa.csv"
    arquivo.write_text(
        "Planetary Systems (PS),,Confirmed Planets (retiring),\n"
        "Database Column Name,Table Label or Comment,Database Column Name,Table Label or Comment\n"
        "Column1,Column2,Column3,Column4\n"
        "pl_name,Planet Name,fpl_name,Planet Name\n"
        "hostname,Host Name,pl_hostname,Host Name\n",
        encoding="utf-8",
    )
    docs = apoio._parse_mapeamento(arquivo)
    assert len(docs) == 2
    assert docs[0].metadata == {"source": "mapa.csv", "row": 0}
    assert "pl_name" in docs[0].page_content
    assert "fpl_name" in docs[0].page_content
    assert "Planetary Systems (PS)" in docs[0].page_content
    assert docs[1].metadata["row"] == 1


def test_parse_mapeamento_aceita_layout_sem_linha_filler(tmp_path):
    arquivo = tmp_path / "mapa2.csv"
    arquivo.write_text(
        "Planetary Systems (PS),,\n"
        "Database Column Name,Table Label or Comment\n"
        "pl_name,Planet Name\n",
        encoding="utf-8",
    )
    docs = apoio._parse_mapeamento(arquivo)
    assert len(docs) == 1
    assert docs[0].metadata["row"] == 0


def test_parse_mapeamento_ignora_linhas_vazias_e_lixo(tmp_path):
    arquivo = tmp_path / "mapa3.csv"
    arquivo.write_text(
        "Tabela,\n"
        "Database Column Name,Table Label or Comment\n"
        "\n"
        "Column1,Column2\n"
        "`,\n"
        "\n",
        encoding="utf-8",
    )
    docs = apoio._parse_mapeamento(arquivo)
    assert docs == []


def test_parse_mapeamento_sem_cabecalho_levanta(tmp_path):
    arquivo = tmp_path / "mal.csv"
    arquivo.write_text("pl_name,Planet Name\n", encoding="utf-8")
    with pytest.raises(ValueError):
        apoio._parse_mapeamento(arquivo)


# ─── tokenização e RecuperadorApoio ──────────────────────────────────────────


def test_tokens_preserva_snake_case_e_remove_stopwords():
    assert apoio._tokens("Na tabela PS, a coluna pl_name?") == [
        "tabela",
        "ps",
        "coluna",
        "pl_name",
    ]


def test_recuperador_apoio_ranking_prioriza_doc_correspondente():
    docs = [
        _doc("O planeta Kepler-186 f foi descoberto no ano de 2014.", "p.txt", 0),
        _doc("na tabela PS a coluna pl_name é descrita como Planet Name.", "m.csv", 1),
        _doc("qualquer outro assunto sem relação com astronomia.", "n.txt", 2),
    ]
    rec = apoio.RecuperadorApoio(docs)
    hits = rec.buscar("em que ano foi descoberto o kepler-186", top_k=3, limiar=0.0)
    assert hits[0][0].metadata["row"] == 0
    assert hits[0][1] > 0


def test_recuperador_apoio_filtra_abaixo_do_limiar():
    docs = [
        _doc("na tabela PS a coluna pl_name é descrita como Planet Name.", "m.csv", 0),
        _doc("na tabela PSCP a coluna fpl_name é descrita como Planet Name.", "m.csv", 1),
        _doc("o planeta kepler-186 f orbita a estrela e foi descoberto em 2014.", "p.txt", 2),
    ]
    rec = apoio.RecuperadorApoio(docs)
    assert rec.buscar("pl_name", top_k=5, limiar=1000.0) == []
    hits = rec.buscar("qual coluna guarda o pl_name", top_k=5, limiar=0.0)
    assert hits and hits[0][0].metadata["row"] == 0


# ─── guard test: corpus real ↔ benchmark 'apoio' ─────────────────────────────


def test_guard_corpus_real_confere_linhas_do_benchmark():
    benchmark = json.loads(
        (RAIZ / "data" / "avaliacao" / "benchmark.json").read_text(encoding="utf-8")
    )
    casos = benchmark.get("apoio", [])
    assert casos, "benchmark precisa ter a seção 'apoio'"

    docs = apoio.carregar_documentos_apoio()
    assert len(docs) > 1000

    por_fonte = {}
    for doc in docs:
        por_fonte.setdefault(doc.metadata["source"], []).append(doc)

    for caso in casos:
        fontes = caso.get("arquivos") or ([caso["arquivo"]] if caso.get("arquivo") else [])
        for fonte in fontes:
            docs_fonte = por_fonte.get(fonte, [])
            assert docs_fonte, f"{fonte} não foi parseado no corpus de apoio"
            for linha in caso["linhas_esperadas"]:
                assert linha < len(docs_fonte), f"{fonte} linha {linha} fora do corpus"
                conteudo = docs_fonte[linha].page_content
                marca = _MARCAS_LINHAS.get((fonte, linha))
                if marca is None:
                    continue
                assert (
                    marca.lower() in conteudo.lower()
                ), f"{fonte}#{linha} deveria conter {marca!r}"


# ─── smoke: a base principal (dataset/vectorstore) não é invalidada ──────────


def test_loading_nao_invalidado_pelo_apoio(monkeypatch, tmp_path):
    from src.tratamento import loading
    from src.tratamento.base_vetorial import ler_fingerprint, salvar_fingerprint

    monkeypatch.setattr(
        "src.tratamento.base_vetorial.CAMINHO_FINGERPRINT", tmp_path / "fingerprint.txt"
    )
    salvar_fingerprint()
    pedacos = loading.obter_pedacos()
    assert pedacos
    fingerprint_atual = loading.calcular_fingerprint_datasets()
    assert fingerprint_atual == ler_fingerprint()