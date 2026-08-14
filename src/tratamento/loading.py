import hashlib
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parents[2]  # raiz do projeto: agente_Ynuyasha
DATASET_DIR = BASE_DIR / "data" / "dataset"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100

DICIONARIO_TIPOS_ASTEROIDES = {
    "an": "asteroide",
}

DICIONARIO_CLASSES_ORBITAIS = {
    "AMO": "Amor",
    "APO": "Apolo",
}

DICIONARIO_TIPOS_EVENTOS = {
    "BH?": "possível buraco negro",
    "Psr": "pulsar",
    "QSO": "quasar",
    "SN*": "supernova",
}

DICIONARIO_TIPOS_SIMBAD = {
    "AGN": "núcleo galáctico ativo",
    "HII": "região H II de formação estelar",
    "LP?": "candidata a anã marrom",
    "OpC": "aglomerado aberto de estrelas",
    "PM*": "estrela com movimento próprio",
    "PN": "nebulosa planetária",
    "SB*": "estrela binária espectroscópica",
    "SNR": "remanescente de supernova",
    "dS*": "estrela anã",
    "s*r": "estrela supergigante",
}

_VALORES_AUSENTES = {"n/a", "nan", "none", "null", "na", ""}


def _limpar(valor) -> str:
    """Formata um valor do CSV para texto legível.

    NaN/vazios viram 'não informado'; números ganham até 6 algarismos
    significativos para leitura natural.
    """
    if valor is None:
        return "não informado"
    if isinstance(valor, float):
        if pd.isna(valor):
            return "não informado"
        if valor.is_integer():
            return str(int(valor))
        return f"{valor:.6g}"
    texto = str(valor).strip()
    if texto.lower() in _VALORES_AUSENTES:
        return "não informado"
    return texto


def _expandir(valor, dicionario) -> str:
    """Expande um código de tipo para texto legível, com o código entre parênteses."""
    chave = str(valor).strip() if valor is not None else None
    nome = dicionario.get(chave)
    if nome:
        return f"{nome} ({chave})"
    return _limpar(valor)


def _template_planetas(linha: dict) -> str:
    return (
        f"O exoplaneta {_limpar(linha.get('nome_planeta'))} orbita a estrela "
        f"{_limpar(linha.get('nome_estrela'))}. Foi descoberto em "
        f"{_limpar(linha.get('ano_descoberta'))} pelo método de "
        f"{_limpar(linha.get('metodo_descoberta'))}. Tem raio de "
        f"{_limpar(linha.get('raio_terrestre'))} vezes o raio da Terra e massa de "
        f"{_limpar(linha.get('massa_terrestre'))} massas terrestres. A temperatura do "
        f"planeta é de {_limpar(linha.get('temperatura_planeta_k'))} K. A estrela "
        f"hospedeira é do tipo espectral {_limpar(linha.get('tipo_espectral_estrela'))}, "
        f"com temperatura de {_limpar(linha.get('temperatura_estrela_k'))} K, raio de "
        f"{_limpar(linha.get('raio_solar_estrela'))} raios solares e massa de "
        f"{_limpar(linha.get('massa_solar_estrela'))} massas solares. O sistema está a "
        f"{_limpar(linha.get('distancia_parsecs'))} parsecs da Terra. "
        f"Fonte: {_limpar(linha.get('fonte_dados'))}."
    )


def _template_habitabilidade(linha: dict) -> str:
    return (
        f"De acordo com {_limpar(linha.get('fonte_dados'))}, o exoplaneta "
        f"{_limpar(linha.get('nome_planeta'))}, que orbita a estrela "
        f"{_limpar(linha.get('nome_estrela'))}, tem raio de "
        f"{_limpar(linha.get('raio_terrestre'))} vezes o raio da Terra, massa de "
        f"{_limpar(linha.get('massa_terrestre'))} massas terrestres, temperatura de "
        f"equilíbrio de {_limpar(linha.get('temperatura_equilibrio_k'))} K e fluxo de "
        f"insolação de {_limpar(linha.get('fluxo_insolacao_terra'))} vezes o da Terra. "
        f"A estimativa de zona habitável é: {_limpar(linha.get('zona_habitavel_estimada'))}."
    )


def _template_glossario(linha: dict) -> str:
    return (
        f"O termo {_limpar(linha.get('termo_cientifico'))} é definido da seguinte forma. "
        f"Definição simples: {_limpar(linha.get('definicao_simples'))} "
        f"Definição técnica: {_limpar(linha.get('definicao_tecnica'))} "
        f"Unidade de medida relacionada: {_limpar(linha.get('unidade_medida_relacionada'))}. "
        f"Fonte: {_limpar(linha.get('fonte_dados'))}."
    )


def _template_constelacoes(linha: dict) -> str:
    return (
        f"A constelação de {_limpar(linha.get('nome_portugues'))} "
        f"(em latim {_limpar(linha.get('nome_latin'))}, sigla "
        f"{_limpar(linha.get('sigla'))}) é visível no hemisfério "
        f"{_limpar(linha.get('hemisferio_visivel'))}. Sua estrela principal é "
        f"{_limpar(linha.get('estrela_principal'))} e possui "
        f"{_limpar(linha.get('quantidade_estrelas_brilhantes'))} estrelas brilhantes. "
        f"Fonte: {_limpar(linha.get('fonte_dados'))}."
    )


def _template_asteroides(linha: dict) -> str:
    return (
        f"O corpo celeste {_limpar(linha.get('nome_corpo'))} é um "
        f"{_expandir(linha.get('tipo_objeto'), DICIONARIO_TIPOS_ASTEROIDES)}, com diâmetro "
        f"de {_limpar(linha.get('diametro_km'))} km. É potencialmente perigoso? "
        f"{_limpar(linha.get('potencialmente_perigoso'))}. É um objeto próximo da Terra? "
        f"{_limpar(linha.get('objeto_proximo_terra'))}. Classe orbital: "
        f"{_expandir(linha.get('classe_orbital'), DICIONARIO_CLASSES_ORBITAIS)}. "
        f"Fonte: {_limpar(linha.get('fonte_dados'))}."
    )


def _template_gaia(linha: dict) -> str:
    return (
        f"A estrela identificada pelo catálogo Gaia DR3 com o id "
        f"{_limpar(linha.get('id_fonte_gaia'))} tem magnitude G de "
        f"{_limpar(linha.get('magnitude_g'))} e paralaxe de "
        f"{_limpar(linha.get('paralaxe_mas'))} milissegundos de arco. Ascensão reta "
        f"{_limpar(linha.get('ascensao_reta_ra'))} graus, declinação "
        f"{_limpar(linha.get('declinacao_dec'))} graus. "
        f"Fonte: {_limpar(linha.get('fonte_dados'))}."
    )


def _paralaxe_texto(valor) -> str:
    paralaxe = _limpar(valor)
    if paralaxe == "não informado":
        return "Paralaxe: não informado."
    return f"Paralaxe: {paralaxe} milissegundos de arco."


def _template_simbad(linha: dict) -> str:
    return (
        f"O objeto astronômico {_limpar(linha.get('identificador_principal'))} é "
        f"classificado como {_expandir(linha.get('tipo_objeto'), DICIONARIO_TIPOS_SIMBAD)}. "
        f"Tipo espectral: {_limpar(linha.get('tipo_espectral'))}. "
        f"{_paralaxe_texto(linha.get('paralaxe_mas'))} Ascensão reta "
        f"{_limpar(linha.get('ascensao_reta_ra'))} graus, declinação "
        f"{_limpar(linha.get('declinacao_dec'))} graus. "
        f"Fonte: {_limpar(linha.get('fonte_dados'))}."
    )


def _template_eventos(linha: dict) -> str:
    return (
        f"O evento astrofísico {_limpar(linha.get('identificador_evento'))} é "
        f"classificado como {_expandir(linha.get('tipo_evento_astrofisico'), DICIONARIO_TIPOS_EVENTOS)}. "
        f"Ascensão reta {_limpar(linha.get('ascensao_reta_ra'))} graus, declinação "
        f"{_limpar(linha.get('declinacao_dec'))} graus. "
        f"Fonte: {_limpar(linha.get('fonte_dados'))}."
    )


CONSTRUTORES = {
    "asteroides_cometas_jpl.csv": _template_asteroides,
    "constelacoes_iau.csv": _template_constelacoes,
    "estrelas_e_objetos_simbad.csv": _template_simbad,
    "estrelas_proximas_gaia.csv": _template_gaia,
    "eventos_transientes_extremos.csv": _template_eventos,
    "glossario_astronomico_conceitos.csv": _template_glossario,
    "habitabilidade_exoplanetas.csv": _template_habitabilidade,
    "planetas_e_estrelas_rag.csv": _template_planetas,
}


def carregar_documentos(diretorio: Path = DATASET_DIR) -> list[Document]:
    """Carrega os CSVs como Documentos com texto semântico por linha.

    Cada linha de um dataset vira um Documento com texto em linguagem natural
    (template por arquivo), facilitando a busca semântica. A metadata preserva
    `source` (arquivo) e `row` (índice 0-based da linha) para as citações.
    """
    documentos = []
    for nome_arquivo, construtor in CONSTRUTORES.items():
        caminho = diretorio / nome_arquivo
        if not caminho.exists():
            print(f"[loading] Aviso: dataset {nome_arquivo} não encontrado; ignorado.")
            continue
        df = pd.read_csv(caminho, encoding="utf-8")
        for indice, linha in df.iterrows():
            texto = construtor(linha.to_dict())
            metadata = {"source": nome_arquivo, "row": int(indice)}
            documentos.append(Document(page_content=texto, metadata=metadata))
    return documentos


def dividir_em_pedacos(documentos: list) -> list:
    """Quebra documentos muito longos, preservando linhas coesas.

    As linhas (com seus templates) são unidades coerentes; o splitter só age em
    textos que excedam CHUNK_SIZE (ex.: definições extensas do glossário).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return text_splitter.split_documents(documentos)


def obter_pedacos() -> list:
    """Carrega os documentos e devolve os pedaços prontos para uso."""
    documentos = carregar_documentos()
    return dividir_em_pedacos(documentos)


def calcular_fingerprint_datasets(diretorio: Path = DATASET_DIR) -> str:
    """Hash sha256 dos CSVs que compõem a base de conhecimento.

    O hash considera exatamente os arquivos usados pela indexação
    (CONSTRUTORES), ordenados de forma determinística. Um arquivo ausente
    contribui com um marcador, então adicionar/remover datasets também
    altera a hash — permitindo detectar índices desatualizados.
    """
    hash_som = hashlib.sha256()
    for nome_arquivo in sorted(CONSTRUTORES):
        caminho = diretorio / nome_arquivo
        hash_som.update(nome_arquivo.encode("utf-8"))
        if caminho.exists():
            hash_som.update(b"presente")
            with caminho.open("rb") as f:
                for bloco in iter(lambda: f.read(8192), b""):
                    hash_som.update(bloco)
        else:
            hash_som.update(b"ausente")
    return hash_som.hexdigest()


if __name__ == "__main__":
    documentos = carregar_documentos()
    pedacos = dividir_em_pedacos(documentos)
    print(f"Documentos carregados: {len(documentos)}")
    print(f"Pedaços gerados: {len(pedacos)}")
    if documentos:
        maior = max(len(d.page_content) for d in documentos)
        print(f"Maior texto de linha: {maior} caracteres")
        print("\nExemplo (planetas):")
        print(documentos[0].page_content)
