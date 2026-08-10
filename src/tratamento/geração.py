import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

CAMINHO_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(CAMINHO_ENV)

MODELO_GROQ = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TEMPERATURA_GROQ = float(os.getenv("GROQ_TEMPERATURE", "0.3"))
MODELO_FALLBACK = os.getenv("OLLAMA_FALLBACK_MODEL", "smollm2:360m")
TEMPERATURA_FALLBACK = 0.3

SISTEMA_ASTRONOMIA = (
    """Você é o Ynuyasha, um agente simpático especializado em astronomia. 
    Responda em português do Brasil, de forma clara e objetiva. 
    SEMPRE responda em Markdown bem organizado e legível no terminal: 
    use títulos (##) para separar seções, negrito nos termos-chave, 
    listas com marcadores para enumerações e tabelas quando houver vários 
    itens ou dados para comparar. Mantenha parágrafos curtos e coesos. 
    Baseie-se APENAS no contexto fornecido; se a informação não estiver no 
    contexto, diga que não encontrou dados sobre isso de maneira sincera. 
    Cite as fontes e linhas indicadas no contexto quando responder, de maneira organizada e coesa. 
    Não invente números, nomes ou fatos, traga somente o que está no contexto."""
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


def montar_prompt(pergunta: str, contexto: str) -> list:
    conteudo_usuario = (
        f"CONTEXTO:\n{contexto}\n\n"
        f"PERGUNTA DO USUÁRIO:\n{pergunta}\n\n"
        "RESPOSTA:"
    )
    return [
        SystemMessage(content=SISTEMA_ASTRONOMIA),
        HumanMessage(content=conteudo_usuario),
    ]


def gerar_resposta(pergunta: str, contexto: str) -> str:
    """Gera a resposta final a partir da pergunta e do contexto recuperado."""
    global _llm, _usando_groq
    prompt = montar_prompt(pergunta, contexto)
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


if __name__ == "__main__":
    import sys

    pergunta = " ".join(sys.argv[1:]) or "O que é um parsec?"
    print(gerar_resposta(pergunta, "Sem contexto disponível para este teste."))
