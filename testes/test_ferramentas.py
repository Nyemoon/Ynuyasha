from src.tratamento import ferramentas
from src.tratamento.ferramentas import (
    ARQUIVO_ASTEROIDES,
    ARQUIVO_CONSTELACOES,
    ARQUIVO_EVENTOS,
    ARQUIVO_GAIA,
    ARQUIVO_GLOSSARIO,
    ARQUIVO_HABITABILIDADE,
    ARQUIVO_PLANETAS,
    ARQUIVO_SIMBAD,
    FERRAMENTAS,
    MAX_LINHAS,
    _ler_csv,
    _normalizar,
    buscar_na_base,
    consultar_asteroides,
    consultar_constelacao,
    consultar_estrelas_gaia,
    consultar_eventos,
    consultar_glossario,
    consultar_habitabilidade,
    consultar_objeto_simbad,
    consultar_planetas,
)


def test_normalizar_colapsa_espacos_e_minusculas():
    assert _normalizar("  M    31  ") == "m 31"


def test_ferramentas_tem_nove_tools():
    assert len(FERRAMENTAS) == 9


def test_ler_csv_carrega_csv_real_e_normaliza():
    linhas = _ler_csv(ARQUIVO_GLOSSARIO)
    assert linhas
    assert "indice" in linhas[0]
    assert "texto" in linhas[0]
    assert "dados" in linhas[0]
    assert linhas[0]["indice"] == 0


def test_ler_csv_arquivo_ausente_retorna_vazio(tmp_path):
    assert _ler_csv("inexistente.csv", diretorio=tmp_path) == []


def test_ler_csv_com_diretorio_fake(tmp_path):
    csv = tmp_path / ARQUIVO_GLOSSARIO
    csv.write_text(
        "termo_cientifico,definicao_simples,definicao_tecnica,unidade_medida_relacionada,fonte_dados\n"
        "parsec,distância grande,paralaxe de um segundo,pc,IAU\n",
        encoding="utf-8",
    )
    linhas = _ler_csv(ARQUIVO_GLOSSARIO, diretorio=tmp_path)
    assert len(linhas) == 1
    assert linhas[0]["indice"] == 0
    assert "parsec" in linhas[0]["texto"]


def test_ler_csv_valores_ausentes_normalizados(tmp_path):
    csv = tmp_path / ARQUIVO_GLOSSARIO
    csv.write_text(
        "termo_cientifico,definicao_simples,definicao_tecnica,unidade_medida_relacionada,fonte_dados\n"
        "termo,,definição,N/A,IAU\n",
        encoding="utf-8",
    )
    linhas = _ler_csv(ARQUIVO_GLOSSARIO, diretorio=tmp_path)
    assert "não informado" in linhas[0]["texto"]


# ─── Ferramentas contra os CSVs reais (offline) ────────────────────────────


def test_consultar_glossario_parsec():
    resultado = consultar_glossario.invoke({"termo": "parsec"})
    assert "Parsec" in resultado
    assert f"Fonte: {ARQUIVO_GLOSSARIO}, Linha" in resultado


def test_consultar_glossario_sem_resultado():
    resultado = consultar_glossario.invoke({"termo": "zzznadaexiste"})
    assert "Nenhum" in resultado
    assert "encontrado" in resultado


def test_consultar_planetas_55_cnc():
    resultado = consultar_planetas.invoke({"termo": "55 Cnc"})
    assert "55 Cnc e" in resultado
    assert "55 Cnc f" in resultado
    assert f"Fonte: {ARQUIVO_PLANETAS}, Linha" in resultado


def test_consultar_habitabilidade_trappist():
    resultado = consultar_habitabilidade.invoke({"termo": "TRAPPIST-1 d"})
    assert "TRAPPIST-1 d" in resultado
    assert "zona habitável" in resultado.lower()
    assert f"Fonte: {ARQUIVO_HABITABILIDADE}, Linha" in resultado


def test_consultar_asteroides_eros():
    resultado = consultar_asteroides.invoke({"termo": "Eros"})
    assert "433 Eros" in resultado
    assert f"Fonte: {ARQUIVO_ASTEROIDES}, Linha" in resultado


def test_consultar_asteroides_potencialmente_perigosos():
    resultado = consultar_asteroides.invoke({"termo": "potencialmente perigosos"})
    assert "1566 Icarus" in resultado
    assert "1620 Geographos" in resultado
    assert "1862 Apollo" in resultado
    assert f"Fonte: {ARQUIVO_ASTEROIDES}, Linha" in resultado


def test_consultar_asteroides_proximos_da_terra():
    resultado = consultar_asteroides.invoke({"termo": "próximos da Terra"})
    assert resultado.count(f"Fonte: {ARQUIVO_ASTEROIDES}, Linha") == MAX_LINHAS
    assert "objeto próximo da terra" in resultado.lower()


def test_consultar_habitabilidade_zona_habitavel_exclui_fora():
    resultado = consultar_habitabilidade.invoke({"termo": "zona habitável"})
    assert "55 Cnc f" in resultado
    assert "51 Peg b" not in resultado
    assert "Fora da Zona Habitável" not in resultado
    assert resultado.count(f"Fonte: {ARQUIVO_HABITABILIDADE}, Linha") == MAX_LINHAS
    assert f"Fonte: {ARQUIVO_HABITABILIDADE}, Linha" in resultado


def test_consultar_constelacao_orion():
    resultado = consultar_constelacao.invoke({"termo": "Ori"})
    assert "Órion" in resultado
    assert f"Fonte: {ARQUIVO_CONSTELACOES}, Linha" in resultado


def test_consultar_objeto_simbad_m31_com_espacos():
    resultado = consultar_objeto_simbad.invoke({"termo": "M 31"})
    assert "M" in resultado
    assert "31" in resultado
    assert f"Fonte: {ARQUIVO_SIMBAD}, Linha" in resultado


def test_consultar_estrelas_gaia_por_id():
    resultado = consultar_estrelas_gaia.invoke({"termo": "5853498713190525696"})
    assert "5853498713190525696" in resultado
    assert f"Fonte: {ARQUIVO_GAIA}, Linha" in resultado


def test_consultar_eventos_quasares():
    resultado = consultar_eventos.invoke({"termo": "QSO"})
    assert "QSO" in resultado
    assert f"Fonte: {ARQUIVO_EVENTOS}, Linha" in resultado


def test_consultar_eventos_respeita_top_n():
    resultado = consultar_eventos.invoke({"termo": "QSO"})
    assert resultado.count(f"Fonte: {ARQUIVO_EVENTOS}, Linha") == MAX_LINHAS


def test_consultar_eventos_sem_resultado():
    resultado = consultar_eventos.invoke({"termo": "zzznadaexiste"})
    assert "Nenhum" in resultado


def test_consultar_planetas_sem_resultado():
    resultado = consultar_planetas.invoke({"termo": "zzznadaexiste"})
    assert "Nenhum" in resultado


# ─── buscar_na_base (com fakes, padrão hermético) ──────────────────────────


class _FakeDocumento:
    def __init__(self, texto, fonte, linha):
        self.page_content = texto
        self.metadata = {"source": fonte, "row": linha}


def test_buscar_na_base_usa_recuperar_contexto(monkeypatch):
    doc = _FakeDocumento("buraco negro em zona habitável", "fake.csv", 3)

    def fake_recuperar(pergunta, k=5):
        assert pergunta == "alguma pergunta"
        assert k == 5
        return [(doc, 0.87)]

    def fake_formatar(resultados):
        return f"contexto formatado: {resultados[0][1]:.3f}"

    monkeypatch.setattr(ferramentas, "recuperar_contexto", fake_recuperar)
    monkeypatch.setattr(ferramentas, "formatar_contexto", fake_formatar)

    resultado = buscar_na_base.invoke({"pergunta": "alguma pergunta"})
    assert resultado == "contexto formatado: 0.870"


def test_buscar_na_base_resultado_vazio(monkeypatch):
    monkeypatch.setattr(ferramentas, "recuperar_contexto", lambda pergunta, k=5: [])
    monkeypatch.setattr(ferramentas, "formatar_contexto", lambda resultados: "")
    assert buscar_na_base.invoke({"pergunta": "fora da base"}) == ""


def test_citacao_inclui_linha_e_arquivo(tmp_path):
    csv = tmp_path / ARQUIVO_GLOSSARIO
    csv.write_text(
        "termo_cientifico,definicao_simples,definicao_tecnica,unidade_medida_relacionada,fonte_dados\n"
        "parsec,distância grande,paralaxe de um segundo,pc,IAU\n"
        "buraco negro,região,horizonte de eventos,M☉,IAU\n",
        encoding="utf-8",
    )
    linhas = _ler_csv(ARQUIVO_GLOSSARIO, diretorio=tmp_path)
    assert linhas[1]["indice"] == 1
    bloco = f"{linhas[1]['texto']}\nFonte: {ARQUIVO_GLOSSARIO}, Linha {linhas[1]['indice']}"
    assert "Fonte: glossario_astronomico_conceitos.csv, Linha 1" in bloco
