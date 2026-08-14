import re
import unicodedata
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from src.tratamento.loading import CONSTRUTORES, DATASET_DIR
from src.tratamento.retrieval import formatar_contexto, recuperar_contexto

MAX_LINHAS = 5  # top-N linhas devolvidas por cada consulta

# Nomes dos arquivos usados pelas ferramentas (mesma convenção do loading).
ARQUIVO_PLANETAS = "planetas_e_estrelas_rag.csv"
ARQUIVO_HABITABILIDADE = "habitabilidade_exoplanetas.csv"
ARQUIVO_ASTEROIDES = "asteroides_cometas_jpl.csv"
ARQUIVO_CONSTELACOES = "constelacoes_iau.csv"
ARQUIVO_GLOSSARIO = "glossario_astronomico_conceitos.csv"
ARQUIVO_SIMBAD = "estrelas_e_objetos_simbad.csv"
ARQUIVO_GAIA = "estrelas_proximas_gaia.csv"
ARQUIVO_EVENTOS = "eventos_transientes_extremos.csv"


def _normalizar(texto) -> str:
    """Minúsculas, sem acentos e com espaços colapsados, para buscas tolerantes.

    Remover diacríticos torna as buscas insensíveis a acentos (ex.: "transito"
    encontra "Método de Trânsito", "orion" encontra "Órion"), cobrindo tanto
    o que o usuário digita quanto o que o LLM emite nos argumentos das ferramentas.
    """
    texto = str(texto or "").strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto)


def _ler_csv(nome: str, diretorio: Path | None = None) -> list[dict]:
    """Lê um dataset, normaliza os valores e gera o texto semântico de cada linha.

    Reutiliza os templates e a normalização de valores do loading.py, garantindo
    que a saída das ferramentas seja consistente com os pedaços da RAG.
    Cada linha vira um dict com:
      - indice: posição 0-based da linha no CSV (usada nas citações);
      - dados:  colunas normalizadas;
      - texto:  texto em linguagem natural (template por arquivo).
    """
    diretorio = diretorio or DATASET_DIR
    caminho = diretorio / nome
    if not caminho.exists():
        return []
    construtor = CONSTRUTORES[nome]
    df = pd.read_csv(caminho, encoding="utf-8")
    linhas = []
    for indice, linha in df.iterrows():
        dados = linha.to_dict()
        linhas.append({"indice": int(indice), "dados": dados, "texto": construtor(dados)})
    return linhas


def _buscar(linhas: list[dict], termo: str, colunas: list[str]) -> list[dict]:
    """Filtra linhas cujo termo apareça (substring) em alguma coluna informada."""
    termo = _normalizar(termo)
    if not termo:
        return []
    return [
        linha
        for linha in linhas
        if any(termo in _normalizar(linha["dados"].get(coluna)) for coluna in colunas)
    ]


def _filtrar_por_valor(linhas: list[dict], campo: str, **regras) -> list[dict]:
    """Filtra por regras de valor (contém/exclui/é igual) sobre uma coluna."""
    resultado = linhas
    if regras.get("contem"):
        resultado = [
            linha
            for linha in resultado
            if regras["contem"] in _normalizar(linha["dados"].get(campo))
        ]
    if regras.get("exclui"):
        resultado = [
            linha
            for linha in resultado
            if regras["exclui"] not in _normalizar(linha["dados"].get(campo))
        ]
    if regras.get("igual") is not None:
        resultado = [
            linha
            for linha in resultado
            if _normalizar(linha["dados"].get(campo)) == regras["igual"]
        ]
    return resultado


def _citar(linha: dict, nome: str) -> str:
    """Formata o resultado de uma linha com a citação Fonte: arquivo, Linha X."""
    return f"{linha['texto']}\nFonte: {nome}, Linha {linha['indice']}"


def _formatar_resultados(resultados: list[dict], nome: str, topico: str) -> str:
    if not resultados:
        return f"Nenhum registro de {topico} encontrado na base de conhecimento."
    blocos = [_citar(linha, nome) for linha in resultados[:MAX_LINHAS]]
    return "\n\n".join(blocos)


@tool
def consultar_planetas(termo: str) -> str:
    """Busca exoplanetas e suas estrelas hospedeiras no catálogo NASA Exoplanet Archive.

    Útil para perguntas sobre raio, massa, temperatura, método de descoberta,
    distância ou estrela hospedeira de um exoplaneta. Pesquisa pelo nome do
    planeta ou da estrela.

    Args:
        termo: nome (ou parte) do planeta ou da estrela, ex.: "TRAPPIST-1".
    """
    linhas = _buscar(
        _ler_csv(ARQUIVO_PLANETAS), termo, ["nome_planeta", "nome_estrela"]
    )
    return _formatar_resultados(linhas, ARQUIVO_PLANETAS, "planetas")


@tool
def consultar_habitabilidade(termo: str) -> str:
    """Busca estimativas de zona habitável de exoplanetas (NASA / PHL).

    Útil para perguntas sobre temperatura de equilíbrio, fluxo de insolação ou
    se um planeta é potencialmente habitável. Pesquisa pelo nome do planeta;
    perguntas sobre "zona habitável" devolvem apenas os planetas potencialmente
    habitáveis.

    Args:
        termo: nome (ou parte) do planeta, ou "zona habitável", ex.: "TRAPPIST-1".
    """
    linhas = _ler_csv(ARQUIVO_HABITABILIDADE)
    termo_norm = _normalizar(termo)
    if any(palavra in termo_norm for palavra in ("habitável", "habitavel", "habitáveis")):
        linhas = _filtrar_por_valor(
            linhas, "zona_habitavel_estimada", exclui="fora"
        )
        return _formatar_resultados(
            linhas, ARQUIVO_HABITABILIDADE, "planetas potencialmente habitáveis"
        )
    linhas = _buscar(linhas, termo, ["nome_planeta", "nome_estrela"])
    return _formatar_resultados(linhas, ARQUIVO_HABITABILIDADE, "habitabilidade")


@tool
def consultar_asteroides(termo: str) -> str:
    """Busca asteroides e cometas próximos da Terra no catálogo NASA JPL.

    Útil para perguntas sobre diâmetro, classe orbital ou se um asteroide é
    potencialmente perigoso. Pesquisa pelo nome do corpo celeste; perguntas sobre
    asteroides "potencialmente perigosos" ou "próximos da Terra" devolvem apenas
    os que atendem ao critério.

    Args:
        termo: nome (ou parte) do asteroide/cometa, ou "potencialmente perigosos",
               ex.: "Eros".
    """
    linhas = _ler_csv(ARQUIVO_ASTEROIDES)
    termo_norm = _normalizar(termo)
    if any(palavra in termo_norm for palavra in ("perigos", "perigoso", "perigosa")):
        linhas = _filtrar_por_valor(
            linhas, "potencialmente_perigoso", igual="sim"
        )
        return _formatar_resultados(
            linhas, ARQUIVO_ASTEROIDES, "asteroides potencialmente perigosos"
        )
    if any(
        palavra in termo_norm
        for palavra in ("proximo", "próximo", "proxima", "próxima", "perto", "near")
    ):
        linhas = _filtrar_por_valor(linhas, "objeto_proximo_terra", igual="sim")
        return _formatar_resultados(
            linhas, ARQUIVO_ASTEROIDES, "asteroides próximos da Terra"
        )
    linhas = _buscar(linhas, termo, ["nome_corpo"])
    return _formatar_resultados(linhas, ARQUIVO_ASTEROIDES, "asteroides e cometas")


@tool
def consultar_constelacao(termo: str) -> str:
    """Busca constelações oficiais (IAU) e suas informações.

    Útil para perguntas sobre hemisfério visível, estrela principal ou número de
    estrelas brilhantes de uma constelação. Pesquisa pelo nome em português,
    latim ou pela sigla oficial.

    Args:
        termo: nome ou sigla da constelação, ex.: "Órion" ou "Ori".
    """
    linhas = _buscar(
        _ler_csv(ARQUIVO_CONSTELACOES),
        termo,
        ["nome_portugues", "nome_latin", "sigla"],
    )
    return _formatar_resultados(linhas, ARQUIVO_CONSTELACOES, "constelações")


@tool
def consultar_glossario(termo: str) -> str:
    """Busca definições de termos científicos de astronomia no glossário (IAU/NASA).

    Útil para perguntas conceituais, ex.: "o que é um parsec?", "explique o método
    de trânsito". Pesquisa pelo termo científico (pode ser parte do termo).

    Args:
        termo: termo ou conceito a pesquisar, ex.: "parsec".
    """
    linhas = _buscar(
        _ler_csv(ARQUIVO_GLOSSARIO), termo, ["termo_cientifico"]
    )
    return _formatar_resultados(linhas, ARQUIVO_GLOSSARIO, "termos do glossário")


@tool
def consultar_objeto_simbad(termo: str) -> str:
    """Busca objetos astronômicos do catálogo SIMBAD (estrelas, nebulosas, galáxias).

    Útil para perguntas sobre o tipo de objeto (quasar, nebulosa, supernova...),
    tipo espectral ou coordenadas de M 31, M 42, Proxima Centauri etc. Pesquisa
    pelo identificador principal do objeto.

    Args:
        termo: identificador (ou parte) do objeto, ex.: "M 31".
    """
    linhas = _buscar(
        _ler_csv(ARQUIVO_SIMBAD), termo, ["identificador_principal"]
    )
    return _formatar_resultados(linhas, ARQUIVO_SIMBAD, "objetos SIMBAD")


@tool
def consultar_estrelas_gaia(termo: str) -> str:
    """Busca estrelas próximas do catálogo ESA Gaia DR3.

    Útil para perguntas sobre magnitude G, paralaxe ou coordenadas de estrelas
    próximas. Pesquisa pelo identificador (id_fonte_gaia) do catálogo Gaia.

    Args:
        termo: id Gaia (ou parte) da estrela, ex.: "5853498713190525696".
    """
    linhas = _buscar(_ler_csv(ARQUIVO_GAIA), termo, ["id_fonte_gaia"])
    return _formatar_resultados(linhas, ARQUIVO_GAIA, "estrelas Gaia")


@tool
def consultar_eventos(termo: str) -> str:
    """Busca eventos astrofísicos extremos (quasares, púlsares, supernovas).

    Útil para perguntas do tipo "liste os quasares", "quais eventos são
    supernovas". Pesquisa pelo identificador do evento ou pelo tipo (QSO, SN*,
    Psr, BH?). Use apenas se a pergunta pedir eventos transientes; para tipos de
    objetos em geral prefira consultar_objeto_simbad.

    Args:
        termo: identificador ou tipo do evento, ex.: "QSO".
    """
    linhas = _buscar(
        _ler_csv(ARQUIVO_EVENTOS),
        termo,
        ["identificador_evento", "tipo_evento_astrofisico"],
    )
    return _formatar_resultados(linhas, ARQUIVO_EVENTOS, "eventos astrofísicos")


@tool
def buscar_na_base(pergunta: str) -> str:
    """Busca livre em toda a base de conhecimento (RAG — semântica + lexical).

    Use quando a pergunta não for bem atendida por uma ferramenta específica de
    dataset ou envolver comparações entre múltiplos datasets. Devolve os trechos
    mais relevantes com fonte, linha e relevância.

    Args:
        pergunta: pergunta em linguagem natural, ex.: "planeta em zona habitável".
    """
    resultados = recuperar_contexto(pergunta, k=5)
    return formatar_contexto(resultados)


FERRAMENTAS = [
    consultar_planetas,
    consultar_habitabilidade,
    consultar_asteroides,
    consultar_constelacao,
    consultar_glossario,
    consultar_objeto_simbad,
    consultar_estrelas_gaia,
    consultar_eventos,
    buscar_na_base,
]


if __name__ == "__main__":
    import sys

    consulta = sys.argv[1] if len(sys.argv) > 1 else "parsec"
    resultado = consultar_glossario.invoke({"termo": consulta})
    print(resultado)
