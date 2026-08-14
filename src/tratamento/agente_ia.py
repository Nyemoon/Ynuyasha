import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.tratamento.ferramentas import FERRAMENTAS
from src.tratamento.geração import (
    MENSAGEM_FORA_DA_BASE,
    MODELO_GROQ,
    SISTEMA_ASTRONOMIA,
    TEMPERATURA_GROQ,
    _historico_para_mensagens,
)

CAMINHO_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(CAMINHO_ENV, override=True)

LIMITE_RECURSAO = 6  # profundidade máxima do loop agente → ferramentas
LIMITE_MENSAGENS_AGENTE = 12  # últimas mensagens enviadas ao LLM (System preservado)
PLACEHOLDER_CHAVE = "sua_chave_aqui"

PROMPT_AGENTE = (
    SISTEMA_ASTRONOMIA
    + """

    7. FERRAMENTAS DISPONÍVEIS: você tem acesso a ferramentas que consultam a base
    de conhecimento em tempo real. É OBRIGATÓRIO chamar pelo menos uma ferramenta
    antes de responder qualquer pergunta. SEMPRE que a pergunta envolver dados concretos
    (nomes de planetas, estrelas, constelações, asteroides, termos do glossário,
    objetos SIMBAD, eventos) chame a ferramenta específica mais adequada antes de
    responder. Se a pergunta for genérica ou comparar vários datasets, use a
    ferramenta buscar_na_base. NÃO responda dados factuais sem antes consultar uma
    ferramenta — elas retornam o conteúdo com a citação no formato
    "Fonte: <arquivo>, Linha <X>". Respostas sem dados retornados por uma ferramenta
    ou sem citação ancorada serão descartadas e substituídas pela recusa.

    8. CITAÇÃO DE FERRAMENTAS: ao usar os dados retornados por uma ferramenta,
    preserve a citação "Fonte: <arquivo>, Linha <X>" que acompanha cada resultado.
    Se a ferramenta não encontrar nada, diga educadamente que a informação não
    consta na base de conhecimento (conforme a regra 4) e NÃO invente dados.

    9. FALLBACK DE FERRAMENTAS: se uma ferramenta específica retornar
    "Nenhum registro ... na base de conhecimento", NÃO conclua ainda que a
    informação não existe. Chame antes a ferramenta buscar_na_base com a
    pergunta original do usuário; se ela também não encontrar nada relevante,
    aí sim diga educadamente que a informação não consta na base (regra 4).
    """
)

_llm_agente = None
_grafo = None


def groq_disponivel() -> bool:
    """True se uma chave Groq válida estiver configurada no ambiente."""
    chave = os.getenv("GROQ_API_KEY", "").strip()
    return bool(chave) and chave != PLACEHOLDER_CHAVE


def _obter_llm() -> ChatGroq:
    """Retorna o LLM com as ferramentas vinculadas, criado uma única vez."""
    global _llm_agente
    if _llm_agente is None:
        _llm_agente = ChatGroq(model=MODELO_GROQ, temperature=TEMPERATURA_GROQ).bind_tools(
            FERRAMENTAS
        )
    return _llm_agente


def _ultimas_mensagens(mensagens: list, limite: int = LIMITE_MENSAGENS_AGENTE) -> list:
    """Mantém o SystemMessage e as últimas `limite` mensagens de conversa.

    O checkpointer continua guardando o histórico completo no sqlite; aqui
    apenas limitamos o que é enviado ao LLM a cada invocação, evitando que
    conversas longas estourem o contexto.
    """
    if len(mensagens) <= limite:
        return mensagens
    sistema = [m for m in mensagens if isinstance(m, SystemMessage)]
    resto = [m for m in mensagens if not isinstance(m, SystemMessage)]
    return sistema + resto[-limite:]


def _montar_grafo(llm, tools, checkpointer=None):
    """Constrói o grafo ReAct: agente → (tools_condition) → ferramentas → agente → END.

    Quando um checkpointer é fornecido, o grafo passa a persistir o estado
    (mensagens) entre invocações, ancorado pelo thread_id do config.
    """

    def _no_agente(state: dict) -> dict:
        return {"messages": [llm.invoke(_ultimas_mensagens(state["messages"]))]}

    grafo = StateGraph(MessagesState)
    grafo.add_node("agente", _no_agente)
    grafo.add_node("ferramentas", ToolNode(tools))
    grafo.add_conditional_edges(
        "agente", tools_condition, {"tools": "ferramentas", "__end__": END}
    )
    grafo.add_edge("ferramentas", "agente")
    grafo.set_entry_point("agente")
    return grafo.compile(checkpointer=checkpointer)


def _criar_grafo(llm=None, tools=None, checkpointer=None):
    """Compila o grafo padrão uma única vez (singleton).

    Quando llm/tools são injetados (ex.: testes) ou há um checkpointer,
    constrói um grafo novo sem interferir no grafo em cache.
    """
    global _grafo
    if checkpointer is None and llm is None and tools is None:
        if _grafo is None:
            _grafo = _montar_grafo(_obter_llm(), FERRAMENTAS)
        return _grafo
    return _montar_grafo(llm or _obter_llm(), tools or FERRAMENTAS, checkpointer)


def _degradar_rag(pergunta: str, historico=None) -> str:
    """Fluxo RAG clássico (recuperação → geração), usado sem chave Groq."""
    from src.tratamento.agente import preparar_contexto
    from src.tratamento.geração import gerar_resposta

    contexto = preparar_contexto(pergunta)
    return gerar_resposta(pergunta, contexto, historico=historico)


def _contar_tool_calls(mensagens: list) -> int:
    """Número de AIMessages com tool_calls na lista."""
    return sum(1 for m in mensagens if isinstance(m, AIMessage) and m.tool_calls)


KEYWORDS_RECUSA = (
    "não consta",
    "nao consta",
    "não encontrei",
    "nao encontrei",
    "não encontrada",
    "nao encontrada",
    "não tenho",
    "nao tenho",
    "nenhum registro",
    "nenhum trecho",
    "não faz parte",
    "nao faz parte",
    "fora da base",
    "fora do meu escopo",
    "base de conhecimento",
    "não posso responder",
    "nao posso responder",
)


def _parece_recusa(texto: str) -> bool:
    """True se o texto parece uma recusa cordial de informação fora da base."""
    baixo = (texto or "").lower()
    return any(palavra in baixo for palavra in KEYWORDS_RECUSA)


def _texto_ferramentas(mensagens: list) -> str:
    """Concatena o conteúdo de todas as ToolMessage (retornos das ferramentas)."""
    return "\n".join(
        m.content for m in mensagens if isinstance(m, ToolMessage)
    )


PADRAO_CITACAO_FERRAMENTA = re.compile(
    r"(?:Fonte|fonte):\s*(?P<arquivo>[^,\n]+?)\s*,\s*(?:Linha|linha)\s*:?\s*(?P<linha>\d+)"
)

# Formato da resposta do LLM: aceita singular ("Linha 5") ou plural
# ("Linhas 5, 7 e 10"), mantendo a exigência de "Fonte:" e "Linha" com
# maiúsculas para que formatos informais continuem fora da ancoragem.
PADRAO_CITACAO_RESPOSTA = re.compile(
    r"Fonte:\s*(?P<arquivo>[^,\n]+?)\s*,\s*Linhas?\s*(?P<numeros>[\d\s,]+(?:e\s*[\d\s,]+)*)"
)


def _extrair_citacoes_resposta(texto: str) -> set[tuple[str, int]]:
    """Extrai (arquivo, linha) das citações da resposta do LLM.

    Suporta "Fonte: X, Linha 5" e "Fonte: X, Linhas 5, 7 e 10" (plural),
    pois modelos frequentemente agrupam as linhas no plural — o que fazia as
    respostas corretas serem descartadas pelo gate de ancoragem.
    """
    citacoes = set()
    for m in PADRAO_CITACAO_RESPOSTA.finditer(texto or ""):
        arquivo = m.group("arquivo").strip()
        numeros = {int(n) for n in re.findall(r"\d+", m.group("numeros"))}
        citacoes.update((arquivo, n) for n in numeros)
    return citacoes


def _extrair_citacoes_ferramenta(texto: str) -> set[tuple[str, int]]:
    """Extrai (arquivo, linha) do retorno das ferramentas, aceitando os dois formatos.

    - Canonical (ferramentas específicas): "Fonte: a.csv, Linha 1"
    - RAG (buscar_na_base / formatar_contexto): "fonte: a.csv, linha: 1"
    """
    return {
        (m.group("arquivo").strip(), int(m.group("linha")))
        for m in PADRAO_CITACAO_FERRAMENTA.finditer(texto or "")
    }


def _verificar_ancoragem(resposta: str, mensagens: list) -> bool:
    """True se a resposta está ancorada nos dados retornados pelas ferramentas.

    1. Com citações na resposta: só passa se TODAS as "Fonte: <arquivo>, Linha X"
       existirem no retorno das ferramentas (bloqueia fontes forjadas).
    2. Sem citações mas com dados devolvidos pelas ferramentas: não ancorada
       (respondeu do conhecimento próprio, ignorando a regra de citação).
    3. Sem dados devolvidos: só passa se a resposta for uma recusa cordial —
       nunca aceita uma resposta factual sem qualquer base.

    O retorno das ferramentas aceita tanto o formato canônico de citação quanto o
    formato RAG ("fonte: X, linha: Y") emitido por `buscar_na_base`.

    A regra de aceitação foi relaxada para não transformar respostas corretas em
    recusa: basta que a resposta contenha **ao menos uma** citação válida (presente
    no retorno das ferramentas) e que **nenhuma** citação seja forjada (isto é,
    nenhuma citação da resposta fora do conjunto retornado pelas ferramentas).
    Respostas sem dados devolvidos continuam exigindo recusa cordial; respostas
    com dados porém sem citação alguma permanecem barradas (responderam do
    conhecimento próprio, ignorando a regra de citação).
    """
    citacoes_resposta = _extrair_citacoes_resposta(resposta)
    citacoes_ferramentas = _extrair_citacoes_ferramenta(_texto_ferramentas(mensagens))
    if citacoes_resposta:
        forjadas = citacoes_resposta - citacoes_ferramentas
        if forjadas:
            return False
        return bool(citacoes_resposta & citacoes_ferramentas)
    if citacoes_ferramentas:
        return False
    return _parece_recusa(resposta)


def _ancorar_resposta(resposta: str, mensagens: list) -> str:
    """Devolve a resposta, ou a recusa padrão quando ela não estiver ancorada."""
    if _verificar_ancoragem(resposta, mensagens):
        return resposta
    return MENSAGEM_FORA_DA_BASE


def _registrar_uso(registrador, resposta: str, tool_calls: int, inicio: float, modo: str) -> None:
    """Entrega os metadados do turno ao callable de observabilidade, se houver."""
    if registrador is None:
        return
    registrador(
        {
            "modo": modo,
            "tool_calls": tool_calls,
            "latencia_s": time.perf_counter() - inicio,
            "citou_fonte": "Fonte:" in resposta,
        }
    )


def executar_agente(
    pergunta: str,
    historico=None,
    llm=None,
    tools=None,
    thread_id=None,
    checkpointer=None,
    registrador=None,
) -> str:
    """Executa o loop de tool-calling e devolve a resposta final do agente.

    Sem chave Groq (e sem llm injetado), degrada para o fluxo RAG clássico
    (thread_id é ignorado nesse caminho).

    Com thread_id, o grafo é compilado com um checkpointer e o estado da
    conversa persiste entre chamadas: o SystemMessage é injetado apenas quando
    o estado ainda está vazio (via get_state). Sem thread_id, o comportamento
    atual é mantido (System + histórico explícito, sem persistência).

    Falhas da Groq (quota esgotada com 429, erro de tool-call 400, instabilidade
    de rede) derrubam automaticamente para o fluxo RAG clássico com Ollama —
    o usuário nunca fica sem resposta por causa da API.

    `registrador` é um callable opcional que recebe metadados do turno
    (modo, tool_calls, latencia_s, citou_fonte) para observabilidade.
    """
    inicio = time.perf_counter()
    if llm is None and not groq_disponivel():
        resposta = _degradar_rag(pergunta, historico)
        _registrar_uso(registrador, resposta, 0, inicio, modo="rag")
        return resposta

    llm = llm or _obter_llm()
    tools = tools or FERRAMENTAS

    try:
        if thread_id is not None:
            from src.tratamento.memoria import obter_memoria

            checkpointer = checkpointer or obter_memoria()
            grafo = _criar_grafo(llm, tools, checkpointer=checkpointer)
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": LIMITE_RECURSAO,
            }
            estado = grafo.get_state(config)
            indice_antes = (
                len(estado.values.get("messages", [])) if estado.values else 0
            )
            mensagens = (
                [SystemMessage(content=PROMPT_AGENTE), HumanMessage(content=pergunta)]
                if not estado.values
                else [HumanMessage(content=pergunta)]
            )
            resultado = grafo.invoke({"messages": mensagens}, config=config)
            mensagens_turno = resultado["messages"][indice_antes:]
            resposta = _ancorar_resposta(mensagens_turno[-1].content, mensagens_turno)
            tool_calls = _contar_tool_calls(mensagens_turno)
            _registrar_uso(registrador, resposta, tool_calls, inicio, modo="agente")
            return resposta

        grafo = _criar_grafo(llm, tools)

        mensagens = [SystemMessage(content=PROMPT_AGENTE)]
        mensagens.extend(_historico_para_mensagens(historico))
        mensagens.append(HumanMessage(content=pergunta))

        resultado = grafo.invoke(
            {"messages": mensagens}, config={"recursion_limit": LIMITE_RECURSAO}
        )
        resposta = _ancorar_resposta(resultado["messages"][-1].content, resultado["messages"])
        tool_calls = _contar_tool_calls(resultado["messages"])
        _registrar_uso(registrador, resposta, tool_calls, inicio, modo="agente")
        return resposta
    except Exception as erro:
        print(
            f"[agente_ia] Groq falhou ({type(erro).__name__}: {erro}); "
            "degradando para RAG com Ollama."
        )
        resposta = _degradar_rag(pergunta, historico)
        _registrar_uso(registrador, resposta, 0, inicio, modo="fallback_ollama")
        return resposta


if __name__ == "__main__":
    import sys

    pergunta = " ".join(sys.argv[1:]) or "Qual a definição de parsec?"
    print(executar_agente(pergunta))
