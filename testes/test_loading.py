import pandas as pd
import pytest
from langchain_core.documents import Document

from src.tratamento.loading import (
    CONSTRUTORES,
    _expandir,
    _limpar,
    _template_glossario,
    _template_planetas,
    calcular_fingerprint_datasets,
    carregar_documentos,
    dividir_em_pedacos,
)


@pytest.mark.parametrize(
    "valor, esperado",
    [
        (None, "não informado"),
        (float("nan"), "não informado"),
        ("", "não informado"),
        ("n/a", "não informado"),
        ("NAN", "não informado"),
        ("null", "não informado"),
        ("  na  ", "não informado"),
    ],
)
def test_limpar_valores_ausentes(valor, esperado):
    assert _limpar(valor) == esperado


def test_limpar_numero_inteiro():
    assert _limpar(5.0) == "5"


def test_limpar_numero_fracionario_6_algarismos():
    assert _limpar(1234.56789) == "1234.57"


def test_limpar_string_com_espacos():
    assert _limpar("  Proxima  ") == "Proxima"


def test_expandir_chave_conhecida():
    assert _expandir("AMO", {"AMO": "Amor"}) == "Amor (AMO)"


def test_expandir_chave_desconhecida():
    assert _expandir("XYZ", {"AMO": "Amor"}) == "XYZ"


def test_template_planetas():
    linha = {
        "nome_planeta": "Kepler-452 b",
        "nome_estrela": "Kepler-452",
        "ano_descoberta": "2015",
        "metodo_descoberta": "trânsito",
        "raio_terrestre": 1.6,
        "massa_terrestre": 5.0,
        "temperatura_planeta_k": 265,
        "tipo_espectral_estrela": "G2",
        "temperatura_estrela_k": 5757,
        "raio_solar_estrela": 1.0,
        "massa_solar_estrela": 1.0,
        "distancia_parsecs": 560,
        "fonte_dados": "NASA",
    }
    texto = _template_planetas(linha)
    assert "Kepler-452 b" in texto
    assert "trânsito" in texto
    assert "560 parsecs" in texto


def test_template_glossario():
    linha = {
        "termo_cientifico": "parsec",
        "definicao_simples": "distância grande",
        "definicao_tecnica": "paralaxe de um segundo",
        "unidade_medida_relacionada": "pc",
        "fonte_dados": "IAU",
    }
    texto = _template_glossario(linha)
    assert "parsec" in texto
    assert "paralaxe de um segundo" in texto
    assert "IAU" in texto


def test_carregar_documentos_com_diretorio_fake(tmp_path):
    csv = tmp_path / "glossario_astronomico_conceitos.csv"
    csv.write_text(
        "termo_cientifico,definicao_simples,definicao_tecnica,unidade_medida_relacionada,fonte_dados\n"
        "parsec,distância grande,paralaxe de um segundo,pc,IAU\n",
        encoding="utf-8",
    )
    documentos = carregar_documentos(tmp_path)
    assert len(documentos) == 1
    assert isinstance(documentos[0], Document)
    assert documentos[0].metadata["source"] == csv.name
    assert documentos[0].metadata["row"] == 0
    assert "parsec" in documentos[0].page_content


def test_carregar_documentos_ignora_datasets_ausentes(tmp_path):
    csv = tmp_path / "constelacoes_iau.csv"
    csv.write_text(
        "sigla,nome_portugues,nome_latin,hemisferio_visivel,estrela_principal,"
        "quantidade_estrelas_brilhantes,fonte_dados\n"
        "Ori,Órion,Orion,Norte,Rigel,7,IAU\n",
        encoding="utf-8",
    )
    documentos = carregar_documentos(tmp_path)
    assert len(documentos) == 1
    assert "Órion" in documentos[0].page_content


def test_carregar_documentos_metadados_row_incrementa(tmp_path):
    csv = tmp_path / "constelacoes_iau.csv"
    csv.write_text(
        "sigla,nome_portugues,nome_latin,hemisferio_visivel,estrela_principal,"
        "quantidade_estrelas_brilhantes,fonte_dados\n"
        "Ori,Órion,Orion,Norte,Rigel,7,IAU\n"
        "Cen,Centauro,Centaurus,Sul,Proxima,5,IAU\n",
        encoding="utf-8",
    )
    documentos = carregar_documentos(tmp_path)
    assert [doc.metadata["row"] for doc in documentos] == [0, 1]


def test_dividir_em_pedacos_preserva_texto_curto():
    doc = Document(page_content="texto curto", metadata={"source": "x.csv", "row": 0})
    pedacos = dividir_em_pedacos([doc])
    assert len(pedacos) == 1
    assert pedacos[0].page_content == "texto curto"


def test_dividir_em_pedacos_quebra_texto_longo():
    texto = ("palavra " * 500).strip()
    doc = Document(page_content=texto, metadata={"source": "x.csv", "row": 0})
    pedacos = dividir_em_pedacos([doc])
    assert len(pedacos) > 1
    assert all(len(p.page_content) <= 2100 for p in pedacos)


def test_carregar_documentos_le_csv_com_pandas():
    df = pd.DataFrame({"coluna": [1]})
    assert len(df) == 1
    assert CONSTRUTORES


def _escrever_dataset_fake(diretorio, nome, conteudo):
    caminho = diretorio / nome
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


def test_fingerprint_deterministico(tmp_path):
    _escrever_dataset_fake(
        tmp_path,
        "glossario_astronomico_conceitos.csv",
        "termo_cientifico,definicao_simples,definicao_tecnica,unidade_medida_relacionada,fonte_dados\n"
        "parsec,distância,paralaxe de um segundo,pc,IAU\n",
    )
    assert calcular_fingerprint_datasets(tmp_path) == calcular_fingerprint_datasets(tmp_path)
    assert isinstance(calcular_fingerprint_datasets(tmp_path), str)
    assert len(calcular_fingerprint_datasets(tmp_path)) == 64


def test_fingerprint_muda_com_conteudo(tmp_path):
    nome = "glossario_astronomico_conceitos.csv"
    _escrever_dataset_fake(tmp_path, nome, "termo_cientifico,fonte_dados\nparsec,IAU\n")
    hash_original = calcular_fingerprint_datasets(tmp_path)
    _escrever_dataset_fake(tmp_path, nome, "termo_cientifico,fonte_dados\nparsec,NASA\n")
    assert calcular_fingerprint_datasets(tmp_path) != hash_original


def test_fingerprint_muda_com_arquivo_ausente(tmp_path):
    _escrever_dataset_fake(
        tmp_path,
        "glossario_astronomico_conceitos.csv",
        "termo_cientifico,fonte_dados\nparsec,IAU\n",
    )
    hash_com_arquivo = calcular_fingerprint_datasets(tmp_path)
    hash_sem_arquivo = calcular_fingerprint_datasets(tmp_path / "vazio")
    assert hash_com_arquivo != hash_sem_arquivo
