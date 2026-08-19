import os
import re
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

CAMINHO_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(CAMINHO_ENV, override=True)

MODELO_GROQ = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TEMPERATURA_GROQ = float(os.getenv("GROQ_TEMPERATURE", "0.3"))
RAZOAMENTO_GROQ = os.getenv("GROQ_REASONING_EFFORT", "low")
MODELO_FALLBACK = os.getenv("OLLAMA_FALLBACK_MODEL", "smollm2:360m")
TEMPERATURA_FALLBACK = 0.3

MENSAGEM_FORA_DA_BASE = (
    "## Informação fora da base de conhecimento\n\n"
    "Peço desculpas, mas essa informação **não consta na minha base de conhecimento** — "
    "trabalho apenas com dados verificados, e esse assunto está fora do meu escopo atual.\n\n"
    "Mas fique tranquilo! Posso ajudar com muito prazer em temas como **exoplanetas**, "
    "**estrelas**, **constelações**, **asteroides**, **cometas**, **zona habitável** e "
    "o **glossário astronômico**. Gostaria de explorar algum desses tópicos?"
)

SOBRE_YNUVASHA = (
    "## Sobre o Ynuyasha\n\n"
    "O Ynuyasha é um agente de inteligência artificial baseado em RAG "
    "(Retrieval-Augmented Generation), especializado em astronomia com alta precisão científica. "
    "Ele responde em português do Brasil de forma simpática, acolhedora, prestativa e "
    "tecnicamente rigorosa, tratando cada usuário com total atenção.\n\n"
    "Ele utiliza exclusivamente uma base de conhecimento local construída a partir de fontes "
    "científicas reais: o NASA Exoplanet Archive, o catálogo Gaia DR3 da ESA, o "
    "SIMBAD/CDS, o banco de dados de pequenos corpos da NASA JPL e a IAU, além de "
    "uma classificação própria de habitabilidade. "
    "Com isso, ele responde com precisão sobre exoplanetas, estrelas, constelações, asteroides, "
    "cometas, zona habitável, glossário astronômico, nebulosas, quasares, púlsares, "
    "supernovas e diversos eventos astrofísicos.\n\n"
    "Ele preza pela verdade científica: nunca inventa números ou fatos; se a informação "
    "não está na base, ele avisa com transparência e educação. Em cada resposta, ele "
    "cita as fontes e as linhas dos dados utilizados, explicando o significado técnico de cada "
    "termo. "
    "Pode ser usado pelo terminal (`python main.py`) ou por uma interface web em "
    "Gradio, utilizando a API Groq com fallback automático para Ollama."
)

# Padrões (sem acentos) usados para reconhecer perguntas sobre o próprio Ynuyasha,
# ex.: "quem é você?", "o que é o Ynuyasha?", "como você funciona?".
_PADROES_SOBRE_SI = [
    re.compile(r"quem\s+e\s+(?:o\s+|a\s+)?(?:voce|ynuyasha)"),
    re.compile(r"o\s+que\s+e\s+(?:o\s+|a\s+)?(?:voce|ynuyasha)"),
    re.compile(r"como\s+funciona\s+(?:o\s+|a\s+)?(?:voce|ynuyasha|agente)"),
    re.compile(
        r"como\s+(?:o\s+|a\s+)?(?:voce|ynuyasha|agente)\s+"
        r"(?:funciona|trabalha|responde|foi\s+(?:criado|feito|desenvolvido|montado)"
        r"|e\s+(?:feito|montado))"
    ),
    re.compile(r"fale\s+(?:sobre|de|a\s+respeito\s+de)\s+(?:o\s+|a\s+)?(?:voce|ynuyasha)"),
    re.compile(r"conte\s+(?:sobre|a\s+respeito\s+de)?\s*(?:o\s+|a\s+)?(?:voce|ynuyasha)"),
    re.compile(
        r"(?:me\s+fale|me\s+conte)\s+(?:sobre|a\s+respeito\s+de)\s+(?:o\s+|a\s+)?(?:voce|ynuyasha)"
    ),
    re.compile(r"sobre\s+(?:o\s+|a\s+)?(?:voce|ynuyasha)"),
    re.compile(r"voce\s+e\s+(?:um|uma|o|a)\s+(?:agente|assistente|ia|bot|robo|programa)"),
    re.compile(r"para\s+que\s+(?:voce|o\s+ynuyasha|o\s+agente)\s+(?:serve|foi\s+criado)\b"),
    re.compile(r"qual\s+(?:e\s+)?(?:a\s+|sua\s+|a\s+sua\s+)(?:funcao|finalidade|missao|papel)"),
    re.compile(r"qual\s+(?:e\s+)?(?:o\s+seu\s+nome|a\s+sua\s+identidade)"),
    re.compile(r"base\s+de\s+conhecimento"),
    re.compile(r"o\s+que\s+(?:ha|tem)\s+(?:na|em)\s+(?:sua|a\s+sua)\s+base"),
    re.compile(r"o\s+que\s+voce\s+(?:sabe(?:\s+fazer)?|tem\s+acesso|pode\s+(?:fazer|responder))[?\s]*$"),
    re.compile(r"quais\s+(?:informacoes|dados|temas|assuntos)\s+(?:voce\s+)?(?:tem|possui|acessa|cobre)[?\s]*$"),
]


def _e_pergunta_sobre_si(pergunta: str) -> bool:
    """Reconhece quando a pergunta é sobre o próprio Ynuyasha (identidade, base, funcionamento)."""
    if not (pergunta or "").strip():
        return False
    texto = unicodedata.normalize("NFD", pergunta)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn").lower()
    return any(padrao.search(texto) for padrao in _PADROES_SOBRE_SI)


SISTEMA_ASTRONOMIA = (
    """Você é o Ynuyasha, um agente simpático, acolhedor e prestativo especializado
    em astronomia e astrofísica rigorosa. Responda em português do Brasil, de forma clara, 
    precisa e descritiva, tratando quem pergunta com respeito, profissionalismo e simpatia.

    Normas de resposta:
    1. Escreva em Markdown bem organizado e legível: use títulos (##) para separar
    seções, negrito nos termos-chave, listas com marcadores para enumerações e
    tabelas quando houver vários itens ou dados para comparar. Mantenha parágrafos
    cortos e coesos.
    2. Baseie-se APENAS no contexto fornecido. Não invente números, nomes ou fatos:
    traga somente o que está no contexto.
    3. Cite as fontes e linhas indicadas no contexto ao responder, de maneira
    organizada e coesa.
    4. Se a informação não estiver no contexto, diga de forma sincera e educada que não
    encontrou dados sobre isso na base de conhecimento e sugira um tema próximo
    em que possa ajudar.
    5. Responda estritamente à pergunta feita. Ignore trechos do contexto que não
    ajudem a responder — não os repita nem os mencione na resposta.
    6. Use o histórico da conversa apenas para entender referências do usuário (ex.:
    "e esse planeta?" se refere ao assunto anterior). NUNCA repita nem re-apresente
    conteúdo de turnos anteriores que não esteja no CONTEXTO atual desta pergunta.
    7. Em perguntas de listagem ou enumeração, liste todos os itens presentes no
    contexto e diga quantos encontrou; não invente itens e não afirme que a lista é
    exaustiva de toda a base de conhecimento.

    Rigor científico e estilo prestativo:
    8. Amplie a descrição técnica de cada informação do contexto com precisão científica absoluta 
    (utilizando conceitos físicos corretos, como período orbital em vez de termos vagos, fluxo normalizado 
    em vez de unidades inadequadas, e a separação correta entre técnicas como trânsito e velocidade radial): 
    para cada número, classificação ou termo técnico citado, acrescente pelo menos uma frase 
    explicando o que significa e por que importa para a pergunta, mantendo total rigor técnico.
    9. Seja proativo, empático e prestativo em ajudar: ao final, ofereça um próximo passo útil baseado na
    pergunta (ex.: "Quer que eu detalhe...?" ou "Posso comparar...?"), sugerindo
    apenas temas presentes na base de conhecimento.
    10. Converse de forma amigável, atenciosa e de alto nível técnico, garantindo autoridade 
    e precisão científica sem perder a suavidade e o acolhimento.
    """
)

_llm = None
_usando_groq = False


def _criar_llm_groq() -> ChatGroq:
    return ChatGroq(
        model=MODELO_GROQ,
        temperature=TEMPERATURA_GROQ,
        reasoning_effort=RAZOAMENTO_GROQ,
    )


def _criar_llm_fallback() -> ChatOllama:
    return ChatOllama(model=MODELO_FALLBACK, temperature=TEMPERATURA_FALLBACK)


def obter_llm():
    """Retorna o modelo de geração, criado uma única vez.

    Prioriza a Groq quando GROQ_API_KEY estiver definida; caso contrário,
    usa o modelo local do Ollama (fallback).
    """
    global _llm, _usando_groq
    if _llm is None:
        if os.getenv("GROQ_API_KEY"):
            print(f"[geração] Usando Groq: {MODELO_GROQ}")
            _llm = _criar_llm_groq()
            _usando_groq = True
        else:
            print(f"[geração] GROQ_API_KEY ausente; usando fallback Ollama: {MODELO_FALLBACK}")
            _llm = _criar_llm_fallback()
            _usando_groq = False
    return _llm


def _historico_para_mensagens(historico) -> list:
    """Converte o histórico (dicts role/content) em mensagens LangChain."""
    mensagens = []
    for item in historico or []:
        papel = str(item.get("role", "")).lower()
        conteudo = str(item.get("content", ""))
        if papel == "user":
            mensagens.append(HumanMessage(content=conteudo))
        elif papel == "assistant":
            mensagens.append(AIMessage(content=conteudo))
    return mensagens


_PADRAO_FONTE_CONTEXTO = re.compile(
    r"fonte:\s*([^,)]+?)\s*,\s*linha:\s*([0-9]+)", re.IGNORECASE
)

# Cabeçalho dos blocos do contexto: "[1] (relevância: 0.73, fonte: a.csv, linha: 7)".
_PADRAO_HEADER_CONTEXTO = re.compile(
    r"\[\s*(\d+)\s*\]\s*\([^)]*fonte:\s*([^,)]+?)\s*,\s*linha:\s*([0-9]+)\)",
    re.IGNORECASE,
)

# Marcadores de citação injetados pela Groq ('【1†linha 0】', '【1†L0-L1】', '【1】').
# São ruído cosmético: a seção '## Fontes' é gerada por código a partir das
# citações que o modelo realmente emite.
_PADRAO_CITACAO_GROQ = re.compile(r"【\s*\d+(?:\s*†[^】]*)?】")
_PADRAO_INDICE_GROQ = re.compile(r"【\s*(\d+)")

# Citação inline escrita pelo próprio modelo ("Fonte: a.csv, linha 7" ou
# "**Fonte:** a.csv, linha 7").
_PADRAO_FONTE_INLINE = re.compile(
    r"(?:\*\*)?[Ff]onte:(?:\*\*)?\s*([^,;]+?)\s*,\s*[Ll]inha\s*(\d+)"
)

# Linha que abre uma seção de fontes manuscrita pelo modelo (ex.: '### Fontes'),
# para ser substituída pela seção gerada por código.
_PADRAO_LINHA_SECAO_FONTES = re.compile(
    r"^(?:#{2,6}[ \t]+fontes?\b|#{1,6}[ \t]+refer[êe]?ncias?\b|\*\*[Ff]ontes?\*\*)[ \t]*:?$",
    re.MULTILINE | re.IGNORECASE,
)


def _extrair_texto_conteudo(conteudo) -> str:
    """Normaliza o conteúdo de uma mensagem para texto simples.

    A Groq pode devolver `content` como string ou como lista de blocos de
    conteúdo (ex.: `[{'text': ..., 'type': 'text'}]`). Este helper garante que o
    pipeline sempre trabalhe com string, extraindo o texto dos blocos `text` e
    ignorando blocos como `reasoning` e `tool_calls`.
    """
    if isinstance(conteudo, str):
        return conteudo
    if isinstance(conteudo, (list, tuple)):
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
    return str(conteudo) if conteudo is not None else ""


def _remover_citacoes_groq(texto: str) -> str:
    """Remove os marcadores de citação '【n†...】' injetados pela Groq."""
    return _PADRAO_CITACAO_GROQ.sub("", texto)


def _indices_citados(texto: str) -> set[int]:
    """Índices de blocos do contexto citados pelo modelo nos marcadores '【n】'."""
    return {int(m.group(1)) for m in _PADRAO_INDICE_GROQ.finditer(texto or "")}


def _fontes_do_contexto(contexto: str) -> dict[int, tuple[str, str]]:
    """Mapa índice do bloco → (fonte, linha) a partir dos cabeçalhos do contexto."""
    return {
        int(m.group(1)): (m.group(2).strip(), m.group(3))
        for m in _PADRAO_HEADER_CONTEXTO.finditer(contexto or "")
    }


def _fontes_inline(texto: str) -> list[tuple[str, str]]:
    """Citações 'Fonte: <arquivo>, linha <N>' escritas inline pelo modelo."""
    return [
        (m.group(1).strip(), m.group(2))
        for m in _PADRAO_FONTE_INLINE.finditer(texto or "")
    ]


def _coletar_fontes(contexto: str) -> list[tuple[str, str]]:
    """Extrai as fontes (arquivo, linha) citadas no bloco de contexto.

    A ordem segue o contexto e duplicatas são removidas, então a seção de
    fontes espelha exatamente o que foi entregue ao modelo.
    """
    fontes = []
    for correspondencia in _PADRAO_FONTE_CONTEXTO.finditer(contexto or ""):
        fonte = correspondencia.group(1).strip()
        linha = correspondencia.group(2)
        chave = (fonte, linha)
        if chave not in fontes:
            fontes.append(chave)
    return fontes


def _selecionar_fontes(
    contexto: str, indices: set[int] | None, corpo: str = ""
) -> list[tuple[str, str]]:
    """Escolhe as fontes da seção '## Fontes' com base no que o modelo citou.

    Prioridade:
    1. Marcadores '【n】' apontando para blocos válidos do contexto;
    2. Citações inline 'Fonte: X, linha N' escritas pelo modelo;
    3. Fallback: todas as fontes do contexto (comportamento original).
    """
    mapa = _fontes_do_contexto(contexto)
    if indices:
        selecionadas = []
        for indice in sorted(indices):
            if indice in mapa and mapa[indice] not in selecionadas:
                selecionadas.append(mapa[indice])
        if selecionadas:
            return selecionadas
    inline = _fontes_inline(corpo or "")
    sem_duplicatas = list(dict.fromkeys(inline))
    if sem_duplicatas:
        return sem_duplicatas
    return _coletar_fontes(contexto)


def _montar_fontes_markdown(
    contexto: str, indices: set[int] | None = None, corpo: str = ""
) -> str:
    """Monta a seção '## Fontes' a partir das fontes selecionadas por código."""
    fontes = _selecionar_fontes(contexto, indices, corpo)
    if not fontes:
        return ""
    linhas = "\n".join(f"- {fonte}, linha {linha}" for fonte, linha in fontes)
    return f"\n\n## Fontes\n\n{linhas}"


def _remover_secao_fontes(texto: str) -> str:
    """Remove uma seção de fontes manuscrita pelo modelo, se existir.

    A seção '## Fontes' é gerada por código a partir das citações reais; uma
    seção manuscrita (ex.: '### Fontes') duplicaria ou contradiria isso.
    """
    linhas = (texto or "").splitlines()
    for i, linha in enumerate(linhas):
        if _PADRAO_LINHA_SECAO_FONTES.match(linha.strip()):
            return "\n".join(linhas[:i]).rstrip()
    return texto


def _titulo_se_necessario(conteudo: str, pergunta: str) -> str | None:
    """Título `#` derivado da pergunta, salvo se a resposta já abrir com título."""
    primeira = next(
        (linha.strip() for linha in (conteudo or "").splitlines() if linha.strip()),
        "",
    )
    if primeira.startswith("#") or not pergunta:
        return None
    titulo = pergunta.strip().rstrip("?")
    return titulo or None


def _aprimorar_markdown(texto: str, contexto: str, pergunta: str = "") -> str:
    """Garante organização Markdown mínima por código: título e seção de fontes.

    O LLM pode não seguir as normas do prompt; este pós-processador impõe um
    título `#` quando a resposta abre com texto corrido, remove seções de fontes
    manuscritas pelo modelo e anexa as fontes reais citadas em '## Fontes'.
    """
    texto = _extrair_texto_conteudo(texto)
    indices = _indices_citados(texto)
    texto = _remover_citacoes_groq(texto).strip()
    if not texto:
        return ""
    titulo = _titulo_se_necessario(texto, pergunta)
    if titulo:
        texto = f"# {titulo}\n\n{texto}"
    texto = _remover_secao_fontes(texto)
    return f"{texto.rstrip()}{_montar_fontes_markdown(contexto, indices, texto)}"


def montar_prompt(pergunta: str, contexto: str, historico=None) -> list:
    if not (contexto or "").strip():
        contexto = "(Nenhum trecho relevante encontrado na base de conhecimento.)"
    conteudo_usuario = (
        f"CONTEXTO:\n{contexto}\n\n"
        f"PERGUNTA DO USUÁRIO:\n{pergunta}\n\n"
        "RESPOSTA:"
    )
    return [
        SystemMessage(content=SISTEMA_ASTRONOMIA),
        *_historico_para_mensagens(historico),
        HumanMessage(content=conteudo_usuario),
    ]


def gerar_resposta(pergunta: str, contexto: str, historico=None, forcar_fallback: bool = False) -> str:
    """Gera a resposta final a partir da pergunta, do contexto e do histórico.

    Sem contexto relevante (pergunta fora da base), recusa sem consultar o LLM:
    o modelo jamais é chamado quando não há dados para ancorar a resposta.
    Com `forcar_fallback`, usa direto o modelo local do Ollama, sem tentar a
    Groq (usado quando o agente já degradou por falha/quota da API).
    """
    if not (contexto or "").strip():
        if _e_pergunta_sobre_si(pergunta):
            contexto = SOBRE_YNUVASHA
        else:
            return MENSAGEM_FORA_DA_BASE
    global _llm, _usando_groq
    prompt = montar_prompt(pergunta, contexto, historico=historico)
    if forcar_fallback:
        llm = _criar_llm_fallback()
        try:
            return _aprimorar_markdown(
                _extrair_texto_conteudo(llm.invoke(prompt).content), contexto, pergunta
            )
        except Exception:
            _llm = llm
            _usando_groq = False
            raise
    llm = obter_llm()
    try:
        resposta = llm.invoke(prompt)
        return _aprimorar_markdown(_extrair_texto_conteudo(resposta.content), contexto, pergunta)
    except Exception as erro:
        if _usando_groq:
            print(f"[geração] Erro na Groq ({erro}); alternando para fallback Ollama.")
            _llm = _criar_llm_fallback()
            _usando_groq = False
            resposta = _llm.invoke(prompt)
            return _aprimorar_markdown(_extrair_texto_conteudo(resposta.content), contexto, pergunta)
        raise


def gerar_resposta_stream(pergunta: str, contexto: str, historico=None):
    """Gera a resposta em streaming, token a token, com fallback Groq→Ollama.

    Contexto vazio (pergunta fora da base) emite a recusa e encerra sem consultar
    o LLM.
    """
    if not (contexto or "").strip():
        if _e_pergunta_sobre_si(pergunta):
            contexto = SOBRE_YNUVASHA
        else:
            yield MENSAGEM_FORA_DA_BASE
            return
    global _llm, _usando_groq
    prompt = montar_prompt(pergunta, contexto, historico=historico)
    llm = obter_llm()
    try:
        for pedaco in _fluxo_com_markdown(llm.stream(prompt), pergunta, contexto):
            yield pedaco
    except Exception as erro:
        if _usando_groq:
            print(f"[geração] Erro na Groq ({erro}); alternando para fallback Ollama.")
            _llm = _criar_llm_fallback()
            _usando_groq = False
            for pedaco in _fluxo_com_markdown(_llm.stream(prompt), pergunta, contexto):
                yield pedaco
        else:
            raise


def _fluxo_com_markdown(gerador, pergunta: str, contexto: str):
    """Emoldura o streaming com o Markdown mínimo por código.

    O título `#` sai no primeiro pedaço e a seção '## Fontes' no último,
    preservando o streaming token a token. Os índices citados pelo modelo
    ('【n】') são acumulados para selecionar as fontes da seção final.
    """
    primeiro = True
    indices = set()
    texto_completo = ""
    for pedaco in gerador:
        conteudo = _extrair_texto_conteudo(
            pedaco.content if hasattr(pedaco, "content") else pedaco
        )
        indices.update(_indices_citados(conteudo))
        conteudo = _remover_citacoes_groq(conteudo)
        if not conteudo:
            continue
        if primeiro:
            primeiro = False
            titulo = _titulo_se_necessario(conteudo, pergunta)
            if titulo:
                yield f"# {titulo}\n\n"
        texto_completo += conteudo
        yield conteudo
    if _PADRAO_LINHA_SECAO_FONTES.search(texto_completo):
        return
    fontes = _montar_fontes_markdown(contexto, indices, texto_completo)
    if fontes:
        yield fontes


if __name__ == "__main__":
    import sys

    pergunta = " ".join(sys.argv[1:]) or "O que é um parsec?"
    print(gerar_resposta(pergunta, "Sem contexto disponível para este teste."))
