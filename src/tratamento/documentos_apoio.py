import os
import re
import csv
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parents[2]  # raiz do projeto: agente_Ynuyasha
APOIO_DIR = BASE_DIR / "data" / "documentos"

# Corpus auxiliar: consultado por BM25 (léxico), sem embeddings e sem tocar na
# vectorstore principal — os datasets de data/dataset seguem intactos.
LIMIAR_APOIO_BM25 = float(os.getenv("RAG_LIMIAR_APOIO_BM25", "8.0"))
NUM_APOIO = 10  # quantos candidatos o apoio devolve antes do filtro de limiar

_PADRAO_CELULA_CODIGO = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")
_PADRAO_FILLER = re.compile(r"^Column\d+$")
_PADRAO_TOKENS = re.compile(r"[a-z0-9_]+")

# Palavras funcionais removidas antes do BM25: o corpus de apoio tem muito texto
# em linguagem natural (código de coluna + comentário) e o ruído das palavras
# comuns derruba o sinal dos termos discriminativos (códigos/planetas).
_STOPWORDS = {
    "o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos",
    "das", "em", "no", "na", "nos", "nas", "e", "ou", "que", "qual", "quais",
    "como", "para", "por", "com", "sobre", "ao", "aos", "à", "às", "é", "são",
    "se", "mais", "menos", "fica", "está", "estão", "ser", "foi", "foram",
    "tem", "têm", "the", "and", "of", "to", "in", "on", "for", "with",
    "from", "at", "by",
}


def _tokens(texto: str) -> list[str]:
    """Tokeniza para BM25: minúsculas, códigos snake_case preservados, sem
    pontuação e sem palavras funcionais (que só adicionam ruído)."""
    return [
        token
        for token in _PADRAO_TOKENS.findall((texto or "").lower())
        if token not in _STOPWORDS
    ]

_FONTES_APOIO = (
    "planetas_validados.txt",
    "ps-conf-ext-mapping.csv",
    "Exoplanet_Archive_Column_Mapping_CSV.csv",
    "old-comp-new-comp-mapping.csv",
    "conf-comp-ext-not-in-ps.csv",
)

_corpus_apoio = None
_recuperador_apoio = None


def _limpar(valor) -> str:
    """Formata um valor cru do arquivo para texto legível."""
    if valor is None:
        return "não informado"
    texto = str(valor).strip()
    if not texto:
        return "não informado"
    return texto


def _eh_linha_vazia(celulas: list[str]) -> bool:
    return all(not celula for celula in celulas)


def _eh_linha_filler(celulas: list[str]) -> bool:
    return all(
        not celula or bool(_PADRAO_FILLER.match(celula)) for celula in celulas
    )


def _eh_codigo_coluna(celula: str) -> bool:
    """True para valores que parecem código de coluna (ou ausência documentada)."""
    if not celula:
        return False
    return celula in ("N/A", "NA") or bool(_PADRAO_CELULA_CODIGO.match(celula))


def _parse_planetas_validados(caminho: Path) -> list[Document]:
    """Lê planetas_validados.txt (linha 1 e 3 são separadores, linha 2 é o
    cabeçalho); cada linha a partir da 4 vira um Documento (row 0-based).
    """
    documentos = []
    with caminho.open(encoding="utf-8") as fh:
        linhas = fh.read().splitlines()

    cabecalho_visto = False
    indice = 0
    for linha in linhas:
        linha = linha.strip()
        if not linha or not linha.strip("="):
            continue
        if not cabecalho_visto:
            cabecalho_visto = True
            continue
        partes = [parte.strip() for parte in linha.split("|")]
        if len(partes) < 3:
            continue
        planeta, estrela, ano = partes[:3]
        texto = (
            f"O planeta {_limpar(planeta)} orbita a estrela {_limpar(estrela)} "
            f"e foi descoberto no ano de {_limpar(ano)}. Essa informação consta "
            f"na lista de planetas confirmados do Exoplanet Archive."
        )
        documentos.append(
            Document(
                page_content=texto,
                metadata={"source": caminho.name, "row": indice},
            )
        )
        indice += 1
    return documentos


def _linha_de_cabecalho(celulas: list[str]) -> bool:
    tem_nome_coluna = any("Column Name" in celula for celula in celulas)
    tem_descricao = any(
        "Label or Comment" in celula or "Description or Comment" in celula
        for celula in celulas
    )
    return tem_nome_coluna and tem_descricao


def _normalizar_tabela(celula: str) -> str:
    tabela = celula.strip()
    return tabela or "tabela não identificada"


def _montar_texto_mapeamento(tabelas: list[str], celulas: list[str]) -> str:
    """Monta frases por par coluna/comentário, liderando pelo código da coluna.

    Sem preâmbulo comum entre documentos: cada Documento começa pela própria
    informação, o que reduz o ruído lexical que o preâmbulo repetido adicionava
    ao BM25 (qualquer pergunta batia nos 300+ documentos de mapeamento).
    """
    frases = []
    for j in range(0, len(celulas), 2):
        codigo = celulas[j] if j < len(celulas) else ""
        comentario = celulas[j + 1] if j + 1 < len(celulas) else ""
        if not codigo and not comentario:
            continue
        tabela = _normalizar_tabela(tabelas[j] if j < len(tabelas) else "")
        if _eh_codigo_coluna(codigo) and comentario:
            frases.append(f"Coluna {codigo} (tabela {tabela}): {comentario}.")
        elif comentario:
            frases.append(f"Tabela {tabela}: {comentario}.")
    if not frases:
        return "Linha sem mapeamento documentado."
    return " ".join(frases)


def _parse_mapeamento(caminho: Path) -> list[Document]:
    """Lê um CSV de mapeamento de colunas do Exoplanet Archive.

    Os arquivos variam de layout (fila de título, cabeçalho, fila ColumnN e
    linhas em branco/lixo no fim), então a leitura é feita com o módulo `csv`
    linha a linha — nunca com pandas, que infla linhas irregulares.
    """
    with caminho.open(encoding="utf-8", newline="") as fh:
        linhas = [linha for linha in csv.reader(fh)]

    indice_cabecalho = None
    for i, linha in enumerate(linhas):
        celulas = [celula.strip() for celula in linha]
        if _linha_de_cabecalho(celulas):
            indice_cabecalho = i
            break
    if indice_cabecalho is None:
        raise ValueError(f"cabeçalho não encontrado em {caminho.name}")

    if indice_cabecalho == 0:
        tabelas = []
    else:
        tabelas = [celula.strip() for celula in linhas[indice_cabecalho - 1]]

    documentos = []
    indice = 0
    for linha in linhas[indice_cabecalho + 1 :]:
        celulas = [celula.strip() for celula in linha]
        if _eh_linha_vazia(celulas) or _eh_linha_filler(celulas):
            continue
        if not any(_eh_codigo_coluna(celula) for celula in celulas):
            continue
        texto = _montar_texto_mapeamento(tabelas, celulas)
        documentos.append(
            Document(
                page_content=texto,
                metadata={"source": caminho.name, "row": indice},
            )
        )
        indice += 1
    return documentos


_CARREGADORES = {
    "planetas_validados.txt": _parse_planetas_validados,
    "ps-conf-ext-mapping.csv": _parse_mapeamento,
    "Exoplanet_Archive_Column_Mapping_CSV.csv": _parse_mapeamento,
    "old-comp-new-comp-mapping.csv": _parse_mapeamento,
    "conf-comp-ext-not-in-ps.csv": _parse_mapeamento,
}


def carregar_documentos_apoio(diretorio: Path = APOIO_DIR) -> list[Document]:
    """Carrega os documentos de apoio (data/documentos) em ordem fixa.

    A `row` é o índice 0-based dentro de cada arquivo após o parser ignorar
    cabeçalhos, separadores, linhas ColumnN e linhas em branco.
    """
    documentos = []
    for nome_arquivo in _FONTES_APOIO:
        caminho = diretorio / nome_arquivo
        if not caminho.exists():
            print(
                f"[documentos_apoio] Aviso: arquivo {nome_arquivo} não encontrado; ignorado."
            )
            continue
        construtor = _CARREGADORES[nome_arquivo]
        documentos.extend(construtor(caminho))
    return documentos


def obter_corpus_apoio() -> list[Document]:
    """Recupera o corpus de apoio em memória, carregando uma única vez."""
    global _corpus_apoio
    if _corpus_apoio is None:
        _corpus_apoio = carregar_documentos_apoio()
    return _corpus_apoio


class RecuperadorApoio:
    """Busca lexical no corpus de apoio via BM25, com pontuação bruta.

    O score é o do BM25Okapi (sem normalização); o limiar LIMIAR_APOIO_BM25
    filtra candidatos fracos para evitar falso-positivos fora da base.
    """

    def __init__(self, documentos):
        self.docs = list(documentos)
        corpus = [_tokens(doc.page_content) for doc in self.docs]
        # k1=1.2 com b=0.6 reduz o viés de comprimento dos textos longos de
        # mapeamento (calibrado sobre os casos reais de data/documentos).
        self.vectorizer = BM25Okapi(corpus, k1=1.2, b=0.6)

    @classmethod
    def from_documents(cls, documentos):
        return cls(documentos)

    def buscar(
        self, pergunta: str, top_k: int = NUM_APOIO, limiar: float = LIMIAR_APOIO_BM25
    ) -> list[tuple[Document, float]]:
        pontuacoes = self.vectorizer.get_scores(_tokens(pergunta))
        pares = sorted(zip(self.docs, pontuacoes), key=lambda par: par[1], reverse=True)
        return [(doc, float(score)) for doc, score in pares if score >= limiar][:top_k]


def obter_recuperador_apoio() -> RecuperadorApoio:
    """Recupera o retriever do corpus de apoio, construído uma única vez."""
    global _recuperador_apoio
    if _recuperador_apoio is None:
        _recuperador_apoio = RecuperadorApoio.from_documents(obter_corpus_apoio())
    return _recuperador_apoio


if __name__ == "__main__":
    documentos = carregar_documentos_apoio()
    print(f"Documentos de apoio carregados: {len(documentos)}")
    por_fonte = {}
    for doc in documentos:
        por_fonte.setdefault(doc.metadata["source"], 0)
        por_fonte[doc.metadata["source"]] += 1
    for nome, total in por_fonte.items():
        print(f"  {nome}: {total}")
    if documentos:
        print("\nExemplo:")
        print(documentos[0].page_content)