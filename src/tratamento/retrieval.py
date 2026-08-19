import os
import re
import unicodedata

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from langchain_core.vectorstores.utils import _cosine_similarity
from rank_bm25 import BM25Okapi

from src.tratamento.base_vetorial import criar_ou_carregar_vectorstore
from src.tratamento.documentos_apoio import (
    LIMIAR_APOIO_BM25,
    NUM_APOIO,
    _tokens,
    obter_recuperador_apoio,
)
from src.tratamento.embeddings import embeddings_model
from src.tratamento.loading import DATASET_DIR, obter_pedacos

_vectorstore = None
_bm25 = None

# Cache do vetor da pergunta: o Gradio chama preparar_contexto para exibir o
# contexto e depois reutiliza o resultado na geração; sem este cache a pergunta
# seria re-embebida (e os candidatos só-BM25 re-escoreados) desnecessariamente.
_ultima_query = None
_ultimo_vetor = None
_ultimo_k = None


def _embedar_pergunta(pergunta: str, k: int) -> list:
    """Embede a pergunta com cache: evita re-embedar a mesma query no mesmo k."""
    global _ultima_query, _ultimo_vetor, _ultimo_k
    if pergunta == _ultima_query and k == _ultimo_k and _ultimo_vetor is not None:
        return _ultimo_vetor
    _ultima_query = pergunta
    _ultimo_k = k
    _ultimo_vetor = embeddings_model.embed_query(pergunta)
    return _ultimo_vetor

LIMIAR_RELEVANCIA = float(os.getenv("RAG_LIMIAR_RELEVANCIA", "0.65"))
# Para perguntas fora da base, o embedding às vezes devolve um trecho fraco mas
# acima do limiar (ex.: "Copa do Mundo" → 0.658 numa constelação). Acima deste
# LIMIAR_FORCA o resultado é aceito; abaixo, a recusa exige que a pergunta
# compartilhe ao menos um token de conteúdo com o trecho (ver _recusar_fraco...).
LIMIAR_FORCA = float(os.getenv("RAG_LIMIAR_FORCA", "0.68"))
# Quantos resultados o BM25 traz antes da fusão RRF.
TOP_CANDIDATOS_BM25 = 10
CONSTANTE_RRF = 60  # constante padrão do Reciprocal Rank Fusion
# Janela da busca vetorial: todos os pedaços já têm vetor gravado na vectorstore,
# então pedir uma janela larga devolve o escore cosseno de tudo em uma única
# chamada — sem re-embedar documentos. O BM25 é fundido por RRF sobre esses
# escores completos, eliminando o antigo re-escore (embed_documents) por query.
TOP_CANDIDATOS_VETORIAL = 500

# Perguntas de enumeração (listas) precisam de mais pedaços para cobrir todos os
# itens relevantes da base (ex.: "Liste planetas descobertos por trânsito").
K_ENUMERACAO = 20
_PADRAO_ENUMERACAO = re.compile(
    r"\b(liste|listar|enumere|quais|quantos|quantas|todos os)\b", re.IGNORECASE
)

# Perguntas de subconjunto-por-atributo (ex.: "Quais asteroides são
# potencialmente perigosos?"). Os trechos de um mesmo dataset são quase idênticos
# entre si no espaço de embeddings (todos pontuam ~0.67), então o ranking não
# separa os que atendem ao atributo — a resposta exata vem de uma consulta
# determinística ao CSV (coluna == valor), sem depender do embedding.
# Comparação: "exato" (igualdade) ou "comeca" (prefixo).
# O 6º campo (coluna_entidade) identifica o objeto da linha (ex.: nome_planeta):
# permite responder fatos sobre um objeto específico (ex.: "O TRAPPIST-1 e está
# na zona habitável?") — caso a pergunta nomeie a entidade, devolve só essa linha.
_FILTROS_ATRIBUTO = [
    (
        re.compile(r"\bzona\s+habit[aá]vel", re.IGNORECASE),
        "habitabilidade_exoplanetas.csv",
        "zona_habitavel_estimada",
        "Potencialmente Habitável",
        "comeca",
        "nome_planeta",
    ),
    (
        re.compile(r"potencialmente\s+perigos?", re.IGNORECASE),
        "asteroides_cometas_jpl.csv",
        "potencialmente_perigoso",
        "Sim",
        "exato",
        "nome_corpo",
    ),
    (
        re.compile(r"pr[oó]xim[ao]s?\s+(da|à|de)\s+(?:a\s+)?[Tt]erra", re.IGNORECASE),
        "asteroides_cometas_jpl.csv",
        "objeto_proximo_terra",
        "Sim",
        "exato",
        "nome_corpo",
    ),
    (
        re.compile(r"quasar(es)?", re.IGNORECASE),
        "eventos_transientes_extremos.csv",
        "tipo_evento_astrofisico",
        "QSO",
        "exato",
        "identificador_evento",
    ),
    (
        re.compile(r"p[úu]lsar(es)?", re.IGNORECASE),
        "eventos_transientes_extremos.csv",
        "tipo_evento_astrofisico",
        "Psr",
        "exato",
        "identificador_evento",
    ),
    (
        re.compile(r"supernova(s)?", re.IGNORECASE),
        "eventos_transientes_extremos.csv",
        "tipo_evento_astrofisico",
        "SN*",
        "exato",
        "identificador_evento",
    ),
    (
        re.compile(
            r"\btr[âa]nsi[st]o\b|m[ée]todo\s+de\s+tr[âa]nsi[st]o",
            re.IGNORECASE,
        ),
        "planetas_e_estrelas_rag.csv",
        "metodo_descoberta",
        "Transit",
        "exato",
        "nome_planeta",
    ),
]

_CACHE_POR_ATRIBUTO: dict[tuple, set[int]] = {}
# Mapa arquivo → [(nome_do_objeto em minúsculas, row)] para fatos por entidade.
_CACHE_ENTIDADES: dict[str, list[tuple[str, int]]] = {}


def k_para_pergunta(pergunta: str, k_padrao: int) -> int:
    """Eleva o k para perguntas de enumeração, mantendo o padrão nos demais casos.

    O gatilho é conservador: verbos imperativos de listagem e interrogativos
    plurais ("quais", "quantos"). Perguntas de fato único ("qual", "o que é",
    "quanto") seguem com o k padrão.
    """
    if _PADRAO_ENUMERACAO.search(pergunta or ""):
        return K_ENUMERACAO
    return k_padrao


def _linhas_correspondentes(
    arquivo: str, coluna: str, valor: str, comparacao: str
) -> set[int]:
    """Índices 0-based das linhas do CSV cuja coluna atende ao valor.

    Consulta determinística ao dataset (não usa embeddings). Resultados são
    cacheados por (arquivo, coluna, valor, comparacao).
    """
    chave = (arquivo, coluna, valor, comparacao)
    if chave in _CACHE_POR_ATRIBUTO:
        return _CACHE_POR_ATRIBUTO[chave]
    df = pd.read_csv(DATASET_DIR / arquivo, encoding="utf-8")
    serie = df[coluna].astype(str)
    if comparacao == "comeca":
        mask = serie.str.startswith(valor)
    elif comparacao == "contem":
        mask = serie.str.contains(valor, regex=False)
    else:
        mask = serie == valor
    linhas = set(df.index[mask].tolist())
    _CACHE_POR_ATRIBUTO[chave] = linhas
    return linhas


def _entidades_do_arquivo(arquivo: str, coluna_entidade: str) -> list[tuple[str, int]]:
    """Lista (nome_do_objeto em minúsculas, row) de um dataset, com cache.

    O nome é reduzido à parte antes do primeiro parêntese (designações como
    "433 Eros (A898 PA)" viram "433 eros"), que é como os usuários nomeiam o
    objeto nas perguntas.
    """
    if arquivo in _CACHE_ENTIDADES:
        return _CACHE_ENTIDADES[arquivo]
    df = pd.read_csv(DATASET_DIR / arquivo, encoding="utf-8")
    entidades = []
    for indice, valor in df[coluna_entidade].items():
        texto = str(valor).strip()
        if not texto:
            continue
        nome = texto.split(" (")[0].split("(")[0].strip().lower()
        if nome:
            entidades.append((nome, int(indice)))
    _CACHE_ENTIDADES[arquivo] = entidades
    return entidades


def _linhas_por_entidade(arquivo: str, coluna_entidade: str, pergunta: str) -> set[int]:
    """Linhas cujo nome do objeto aparece no texto da pergunta.

    Ex.: "O TRAPPIST-1 e está na zona habitável?" → linha do "TRAPPIST-1 e".
    A busca vetorial não separa as linhas quase idênticas de um mesmo dataset;
    a menção explícita do objeto é o sinal mais forte disponível.
    """
    texto = (pergunta or "").lower()
    return {
        linha
        for nome, linha in _entidades_do_arquivo(arquivo, coluna_entidade)
        if nome and nome in texto
    }


def _recuperar_exatos_por_atributo(
    pergunta: str,
) -> list[tuple[Document, float]] | None:
    """Recupera exatamente os trechos que respondem a uma pergunta de atributo.

    Duas modalidades, ambas determinísticas (sem embeddings):
    1. Enumeração ("Quais X são Y?"): devolve todas as linhas cujo valor de
       coluna atende ao atributo (ex.: todos os planetas na zona habitável).
    2. Fato sobre um objeto ("O X está na zona habitável?"): devolve apenas as
       linhas cujo nome do objeto aparece na pergunta (ex.: só o TRAPPIST-1 e),
       independente do valor do atributo — o modelo lê o dado e responde.

    Devolve:
    - None: a pergunta não casa com nenhum padrão/entidade → usar o ranking normal;
    - lista: os trechos exatos (com relevância 1.0), possivelmente vazia.
    """
    eh_enumerecao = bool(_PADRAO_ENUMERACAO.search(pergunta or ""))
    for padrao, arquivo, coluna, valor, comparacao, coluna_entidade in _FILTROS_ATRIBUTO:
        if not padrao.search(pergunta):
            continue
        if eh_enumerecao:
            linhas = _linhas_correspondentes(arquivo, coluna, valor, comparacao)
        else:
            linhas = _linhas_por_entidade(arquivo, coluna_entidade, pergunta)
            if not linhas:
                return None  # sem objeto nomeado → ranking normal
        if not linhas:
            return []
        vistos = set()
        exatos = []
        for doc in obter_pedacos():
            if doc.metadata.get("source") != arquivo:
                continue
            linha = doc.metadata.get("row")
            if linha is None or int(linha) not in linhas:
                continue
            chave = _chave_documento(doc)
            if chave in vistos:
                continue
            vistos.add(chave)
            exatos.append((doc, 1.0))
        return exatos
    return None

# Apoio (data/documentos): quantos itens do corpus auxiliar acompanham a base
# principal no contexto, no máximo.
MAX_APOIO_CONTEXTO = 2


def _obter_vectorstore():
    """Recupera a vectorstore em memória, carregando uma única vez."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = criar_ou_carregar_vectorstore()
    return _vectorstore


class BM25Recuperador:
    """Retriever léxico BM25 local, substituto do BM25Retriever do
    langchain-community (que está em sunset).

    Reproduz o comportamento original: tokenização por quebra de espaços,
    BM25Okapi e `.invoke()` devolvendo os `k` documentos mais relevantes.
    """

    def __init__(self, documentos, vectorizer):
        self.docs = list(documentos)
        self.vectorizer = vectorizer
        self.k = TOP_CANDIDATOS_BM25

    @classmethod
    def from_documents(cls, documentos):
        corpus = [doc.page_content.split() for doc in documentos]
        return cls(documentos, BM25Okapi(corpus))

    def invoke(self, pergunta):
        return self.vectorizer.get_top_n(pergunta.split(), self.docs, n=self.k)


def _obter_bm25() -> BM25Recuperador:
    """Recupera o retriever lexical BM25, construído uma única vez.

    Usa os mesmos pedaços (com templates semânticos) da Fase de ingestão.
    """
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Recuperador.from_documents(obter_pedacos())
    return _bm25


def _chave_documento(doc: Document) -> tuple:
    """Identidade estável de um Documento para a fusão entre recuperadores.

    Prioriza (source, row) — a mesma convenção usada nas citações — e recorre
    ao hash do conteúdo quando os metadados não estiverem disponíveis.
    """
    fonte = doc.metadata.get("source")
    linha = doc.metadata.get("row")
    if fonte and linha is not None:
        return (str(fonte), int(linha))
    return ("conteudo", hash(doc.page_content))


def _cosseno(vetor_a: list[float], vetor_b: list[float]) -> float:
    """Similaridade cosseno entre dois vetores.

    Reutiliza a mesma função da vectorstore (InMemoryVectorStore), garantindo
    que o score manual esteja na mesma escala do limiar de relevância.
    """
    matriz = _cosine_similarity(np.array([vetor_a]), np.array([vetor_b]))
    return float(matriz[0][0])


def filtrar_por_relevancia(
    resultados: list[tuple[Document, float]], limiar: float
) -> list[tuple[Document, float]]:
    """Mantém apenas (doc, score) com score >= limiar."""
    return [(doc, score) for doc, score in resultados if score >= limiar]


def _recusar_fraco_sem_sobreposicao(
    pergunta: str, resultados: list[tuple[Document, float]]
) -> bool:
    """Detecta trecho recuperado sem respaldo: fraco e lexicalmente desconexo.

    A recusa honesta não pode depender só do limiar cosseno: perguntas fora da
    base às vezes "encaixam" em um trecho de astronomia com escore fraco
    (0.65–0.68). Nesses casos exigimos que a pergunta e o trecho compartilhem ao
    menos um token de conteúdo (tokenização com stopwords, _tokens de
    documentos_apoio). Sem nenhum token em comum, o trecho é considerado ruído e
    a pergunta é tratada como fora da base (resultado vazio → recusa).
    """
    if not resultados:
        return False
    if resultados[0][1] >= LIMIAR_FORCA:
        return False
    tokens_pergunta = set(_tokens(pergunta or ""))
    tokens_documento = set()
    for doc, _score in resultados[:3]:
        tokens_documento |= set(_tokens(doc.page_content))
    return not (tokens_pergunta & tokens_documento)


def _fusao_rrf(
    listas: list[list[Document]], constante: int = CONSTANTE_RRF
) -> list[Document]:
    """Funde listas ordenadas de Documentos via Reciprocal Rank Fusion (RRF).

    Documentos que aparecem bem posicionados em várias listas sobem no ranking
    final. A identidade de cada Documento é dada por _chave_documento.
    """
    soma = {}
    docs_por_chave = {}
    for lista in listas:
        for posicao, doc in enumerate(lista):
            chave = _chave_documento(doc)
            soma[chave] = soma.get(chave, 0.0) + 1.0 / (constante + posicao + 1)
            docs_por_chave.setdefault(chave, doc)
    ordenados = sorted(soma.items(), key=lambda item: item[1], reverse=True)
    return [docs_por_chave[chave] for chave, _ in ordenados]


# ─── Retrieval conversacional ────────────────────────────────────────────────
# Follow-ups ("e esse planeta?", "e quanto vale em anos-luz?") referenciam o
# turno anterior, mas o embedding/BM25 só viam a pergunta atual — o assunto da
# vez sumia da busca. Aqui expandimos a QUERY DE BUSCA com as últimas trocas
# (apenas quando a pergunta atual parece referenciar o turno anterior), sem
# tocar na pergunta original usada na geração.
MAX_ITENS_CONTEXTO_BUSCA = 4  # últimas mensagens (2 trocas pergunta+resposta)
MAX_TAMANHO_ITEM_CONTEXTO = 200  # caracteres por mensagem anexada
MAX_TAMANHO_CONTEXTO_BUSCA = 600  # teto do trecho anexado à query
# Dêiticos/pronomes que apontam ao turno anterior.
_PALAVRAS_REFERENCIA = frozenset({
    "esse", "esta", "este", "esses", "estas", "estes",
    "disso", "nisso", "nesse", "nessa", "neste", "nesta",
    "aquela", "aquele", "aquelas", "aqueles", "naquele", "naquela",
    "ele", "ela", "eles", "elas", "nele", "nela", "neles", "nelas",
    "dele", "dela", "deles", "delas", "seu", "sua", "seus", "suas",
    "anterior", "anteriormente", "acima", "citado", "citada",
    "citados", "citadas", "mencionado", "mencionada", "mencionados",
    "mencionadas", "tambem", "mesmo", "mesma", "mesmos", "mesmas",
})
# Continuadores típicos de follow-up em pt-BR ("e quanto vale?", "mas e aí?").
_CONECTORES_CONTINUACAO = frozenset({
    "e", "mas", "entao", "porem", "tambem", "afinal", "ai", "nem",
})
_PADRAO_TOKEN_CONTEXTO = re.compile(r"[a-zA-Z0-9_]+")


def _sem_acentos(texto: str) -> str:
    """Remove acentos e converte para minúsculas (padrão pt-BR da UI)."""
    decomposto = unicodedata.normalize("NFD", (texto or "").lower())
    return "".join(c for c in decomposto if unicodedata.category(c) != "Mn")


def _texto_mensagem(msg) -> str:
    """Extrai texto plano de uma mensagem do histórico (str ou dict de role/content).

    Tolerante a conteúdo do Gradio em lista de blocos: ``[{"type": "text", ...}]``.
    """
    if isinstance(msg, str):
        return msg
    if not isinstance(msg, dict):
        return str(msg or "")
    conteudo = msg.get("content", "")
    if isinstance(conteudo, list):
        partes = []
        for bloco in conteudo:
            if isinstance(bloco, str):
                partes.append(bloco)
            elif isinstance(bloco, dict):
                tipo = bloco.get("type")
                texto = bloco.get("text")
                if tipo in (None, "text", "output_text") and texto:
                    partes.append(str(texto))
        return "".join(partes)
    return str(conteudo or "")


def _eh_referencia_ao_turno_anterior(pergunta: str) -> bool:
    """True se a pergunta atual referencia o turno anterior.

    Sinal 1: dêiticos/pronomes ("esse", "dele", "isso"...). Sinal 2: começa com
    um continuador ("e...", "mas...", "também..."). Perguntas autossuficientes
    ("O que é a matéria escura?") não casam nenhum dos dois e preservam o pivô.
    """
    texto = _sem_acentos(pergunta or "")
    tokens = _PADRAO_TOKEN_CONTEXTO.findall(texto)
    if not tokens:
        return False
    return tokens[0] in _CONECTORES_CONTINUACAO or bool(set(tokens) & _PALAVRAS_REFERENCIA)


def _query_com_contexto(pergunta: str, historico=None) -> str:
    """Expande a query de busca com as últimas trocas quando for referência.

    Regras:
    - Sem histórico ou sem indício de referência (mudança de assunto autossuficiente)
      → devolve a pergunta original, preservando pivôs.
    - Com referência → anexa a última pergunta e a cauda da resposta, com limites
      de itens e caracteres, separadas por " | ".
    """
    historico = list(historico or [])
    if not historico:
        return pergunta
    if not _eh_referencia_ao_turno_anterior(pergunta):
        return pergunta
    trechos = []
    for msg in historico[-MAX_ITENS_CONTEXTO_BUSCA:]:
        texto = _texto_mensagem(msg).strip()
        if texto:
            trechos.append(texto[:MAX_TAMANHO_ITEM_CONTEXTO])
    if not trechos:
        return pergunta
    contexto_busca = " | ".join(trechos)[:MAX_TAMANHO_CONTEXTO_BUSCA]
    return f"{pergunta.strip()} | {contexto_busca}"


def recuperar_contexto(
    pergunta: str,
    k: int = 5,
    historico=None,
    vectorstore=None,
    bm25: BM25Recuperador | None = None,
) -> list[tuple[Document, float]]:
    """Busca os trechos mais relevantes combinando busca semântica e lexical.

    1. Escoreia a vectorstore inteira de uma vez (janela larga sobre os vetores
       já gravados, sem re-embedar documentos).
    2. Recupera o topo do BM25.
    3. Funde os dois rankings por RRF.
    4. Mantém os trechos acima do limiar de relevância e devolve os k melhores.

    Como todos os pedaços da base estão na vectorstore, cada candidato do BM25
    já possui escore cosseno — o re-escore por embeddings deixou de ser
    necessário. Se nenhum trecho atingir o limiar, retorna lista vazia —
    indicando que a pergunta não consta na base de conhecimento.

    Perguntas de subconjunto-por-atributo ("Quais X são Y?") passam por um
    caminho determinístico que devolve exatamente as linhas do CSV que atendem
    ao atributo pedido, sem depender do ranking vetorial (ver
    _recuperar_exatos_por_atributo).

    `historico` (lista de dicts role/content) expande a query de busca quando a
    pergunta atual referencia o turno anterior (retrieval conversacional); a
    pergunta original segue intacta para o gate de recusa.
    """
    exatos = _recuperar_exatos_por_atributo(pergunta)
    if exatos is not None:
        return exatos

    vs = vectorstore or _obter_vectorstore()
    bm = bm25 or _obter_bm25()

    # Follow-ups expandem a query de busca com o turno anterior; a pergunta
    # original segue intacta para o gate de recusa e para a geração.
    pergunta_busca = _query_com_contexto(pergunta, historico)

    # Quando o store aceita busca por vetor pré-computado, embute a pergunta uma
    # única vez e reaproveita o vetor também para os candidatos do BM25 (evita o
    # segundo embedding da pergunta no caminho mais lento).
    usa_vetor_unico = hasattr(vs, "similarity_search_with_score_by_vector")
    vetor_pergunta = None
    if usa_vetor_unico:
        vetor_pergunta = _embedar_pergunta(pergunta_busca, k)
        candidatos_vetorial = vs.similarity_search_with_score_by_vector(
            vetor_pergunta, k=TOP_CANDIDATOS_VETORIAL
        )
    else:
        candidatos_vetorial = vs.similarity_search_with_score(
            pergunta_busca, k=TOP_CANDIDATOS_VETORIAL
        )
    candidatos_bm25 = bm.invoke(pergunta_busca)

    fundidos = _fusao_rrf(
        [[doc for doc, _ in candidatos_vetorial], candidatos_bm25]
    )

    score_por_chave = {
        _chave_documento(doc): score for doc, score in candidatos_vetorial
    }

    resultados = [
        (doc, score_por_chave.get(_chave_documento(doc), 0.0)) for doc in fundidos
    ]
    filtrados = filtrar_por_relevancia(resultados, LIMIAR_RELEVANCIA)[:k]
    if _recusar_fraco_sem_sobreposicao(pergunta, filtrados):
        return []
    return filtrados


def formatar_contexto(resultados: list[tuple[Document, float]]) -> str:
    """Monta o texto do contexto (conteúdo + fonte) para injetar no prompt."""
    blocos = []
    for i, (doc, score) in enumerate(resultados, start=1):
        fonte = doc.metadata.get("source", "desconhecida")
        linha = doc.metadata.get("row", "?")
        blocos.append(
            f"[{i}] (relevância: {score:.3f}, fonte: {fonte}, linha: {linha})\n{doc.page_content}"
        )
    return "\n\n".join(blocos)


def recuperar_contexto_formatado(pergunta: str, k: int = 5) -> str:
    """Atalho: recupera e formata o contexto em uma única chamada."""
    return formatar_contexto(recuperar_contexto(pergunta, k=k))


def _recuperar_apoio(pergunta: str, recuperador_apoio=None) -> list:
    """Busca candidatos no corpus de apoio, já filtrados pelo limiar BM25."""
    recuperador = recuperador_apoio or obter_recuperador_apoio()
    return recuperador.buscar(pergunta, top_k=NUM_APOIO, limiar=LIMIAR_APOIO_BM25)


def recuperar_contexto_com_apoio(
    pergunta: str,
    k: int = 5,
    historico=None,
    vectorstore=None,
    bm25: BM25Recuperador | None = None,
    recuperador_apoio=None,
) -> list[tuple[Document, float]]:
    """Recupera o contexto da base principal e o complementa com o apoio.

    1. Busca na base principal (recuperar_contexto com o histórico conversacional).
    2. Busca no corpus de apoio (BM25 léxico) com limiar próprio — usando a
       pergunta original, sem o contexto do histórico.
    3. Anexa até MAX_APOIO_CONTEXTO itens de apoio que não repitam documentos
       da base principal, com prioridade para o contexto principal (o apoio
       nunca é citado quando a base principal já cobre o assunto).
    """
    principal = recuperar_contexto(
        pergunta, k=k, historico=historico, vectorstore=vectorstore, bm25=bm25
    )
    candidatos_apoio = _recuperar_apoio(pergunta, recuperador_apoio)

    vistos = {_chave_documento(doc) for doc, _ in principal}
    apoio_escolhidos = []
    for doc, score in candidatos_apoio:
        chave = _chave_documento(doc)
        if chave in vistos:
            continue
        apoio_escolhidos.append((doc, score))
        vistos.add(chave)
        if len(apoio_escolhidos) >= MAX_APOIO_CONTEXTO:
            break
    return principal + apoio_escolhidos


if __name__ == "__main__":
    import sys

    pergunta = " ".join(sys.argv[1:]) or "planeta em zona habitável"
    resultados = recuperar_contexto(pergunta)
    print(
        f"{len(resultados)} pedaços recuperados para: {pergunta!r} "
        f"(limiar: {LIMIAR_RELEVANCIA})\n"
    )
    texto = formatar_contexto(resultados)
    if texto:
        print(texto)
    else:
        print("Nenhum trecho relevante encontrado na base de conhecimento.")
