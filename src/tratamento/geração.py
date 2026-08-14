import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

CAMINHO_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(CAMINHO_ENV, override=True)

MODELO_GROQ = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TEMPERATURA_GROQ = float(os.getenv("GROQ_TEMPERATURE", "0.3"))
MODELO_FALLBACK = os.getenv("OLLAMA_FALLBACK_MODEL", "smollm2:360m")
TEMPERATURA_FALLBACK = 0.3

MENSAGEM_FORA_DA_BASE = (
    "## Informação fora da base de conhecimento\n\n"
    "Desculpe, essa informação **não consta na minha base de conhecimento** — eu só "
    "respondo com base nos dados que tenho, e essa pergunta está fora do meu escopo.\n\n"
    "Posso ajudar com temas como **exoplanetas**, **estrelas**, **constelações**, "
    "**asteroides**, **cometas**, **zona habitável** e **glossário astronômico**. "
    "Que tal me perguntar algo sobre um desses assuntos?"
)

SISTEMA_ASTRONOMIA = (
    """Você é o Ynuyasha, um agente simpático especializado em astronomia. 
    Responda em português do Brasil, de forma clara e objetiva.
    SEMPRE responda em Markdown bem organizado e legível no terminal: 
    -- use títulos (##) para separar seções, negrito nos termos-chave,
    -- listas com marcadores para enumerações e tabelas quando houver vários
    -- itens ou dados para comparar. Mantenha parágrafos curtos e coesos.
    -- Baseie-se APENAS no contexto fornecido; se a informação não estiver no
    -- contexto, diga que não encontrou dados sobre isso de maneira sincera.
    -- Cite as fontes e linhas indicadas no contexto quando responder, de maneira organizada e coesa.
    -- Não invente números, nomes ou fatos, traga somente o que está no contexto.

    2. CONVERSA ANTERIOR: Use as mensagens anteriores apenas para interpretar referências
    do usuário (ex.: "e esse planeta?" se refere ao assunto anterior). Responda somente com
    base no CONTEXTO fornecido — não repita nem invente fatos que não estejam no CONTEXTO.
    """
)

_llm = None
_usando_groq = False


def _criar_llm_groq() -> ChatGroq:
    return ChatGroq(model=MODELO_GROQ, temperature=TEMPERATURA_GROQ)


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
        return MENSAGEM_FORA_DA_BASE
    global _llm, _usando_groq
    prompt = montar_prompt(pergunta, contexto, historico=historico)
    if forcar_fallback:
        llm = _criar_llm_fallback()
        try:
            return llm.invoke(prompt).content
        except Exception:
            _llm = llm
            _usando_groq = False
            raise
    llm = obter_llm()
    try:
        resposta = llm.invoke(prompt)
        return resposta.content
    except Exception as erro:
        if _usando_groq:
            print(f"[geração] Erro na Groq ({erro}); alternando para fallback Ollama.")
            _llm = _criar_llm_fallback()
            _usando_groq = False
            resposta = _llm.invoke(prompt)
            return resposta.content
        raise


def gerar_resposta_stream(pergunta: str, contexto: str, historico=None):
    """Gera a resposta em streaming, token a token, com fallback Groq→Ollama.

    Contexto vazio (pergunta fora da base) emite a recusa e encerra sem consultar
    o LLM.
    """
    if not (contexto or "").strip():
        yield MENSAGEM_FORA_DA_BASE
        return
    global _llm, _usando_groq
    prompt = montar_prompt(pergunta, contexto, historico=historico)
    llm = obter_llm()
    try:
        for pedaco in llm.stream(prompt):
            yield pedaco.content
    except Exception as erro:
        if _usando_groq:
            print(f"[geração] Erro na Groq ({erro}); alternando para fallback Ollama.")
            _llm = _criar_llm_fallback()
            _usando_groq = False
            for pedaco in _llm.stream(prompt):
                yield pedaco.content
        else:
            raise


if __name__ == "__main__":
    import sys

    pergunta = " ".join(sys.argv[1:]) or "O que é um parsec?"
    print(gerar_resposta(pergunta, "Sem contexto disponível para este teste."))
