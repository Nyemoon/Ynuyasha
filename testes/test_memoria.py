"""Testes herméticos da memória persistente do agente (Fase H).

Usa MemorySaver + FakeLLM (sem Groq/Ollama). Nunca chama `obter_memoria()`
sem checkpointer injetado, evitando criar o arquivo SQLite durante os testes.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from src.tratamento import agente_ia, ferramentas, memoria


class FakeLLM:
    """Responde texto diretamente (sem tool_calls) e registra as invocações."""

    def __init__(self, resposta="resposta final"):
        self.resposta = resposta
        self.invokes = []

    def invoke(self, mensagens):
        self.invokes.append(list(mensagens))
        return AIMessage(content=self.resposta)


@pytest.fixture(autouse=True)
def reseta_globais():
    yield
    memoria._memoria = None
    agente_ia._llm_agente = None
    agente_ia._grafo = None


# ─── singleton e ids de sessão ───────────────────────────────────────────────


def test_novo_thread_id_unico():
    a = memoria.novo_thread_id()
    b = memoria.novo_thread_id()
    assert a != b
    assert len(a) == 32


def test_obter_memoria_devolve_checkpointer_injetado():
    cp = MemorySaver()
    assert memoria.obter_memoria(checkpointer=cp) is cp


# ─── persistência por thread ─────────────────────────────────────────────────


def _executar(pergunta, llm, checkpointer, thread_id):
    return agente_ia.executar_agente(
        pergunta,
        llm=llm,
        tools=ferramentas.FERRAMENTAS,
        checkpointer=checkpointer,
        thread_id=thread_id,
    )


def test_mesmo_thread_acumula_historico():
    llm = FakeLLM()
    cp = MemorySaver()
    thread = "sessao_1"

    _executar("primeira pergunta", llm, cp, thread)
    _executar("segunda pergunta", llm, cp, thread)

    primeira = llm.invokes[0]
    segunda = llm.invokes[1]
    assert isinstance(primeira[0], SystemMessage)
    assert primeira[-1].content == "primeira pergunta"
    assert len(segunda) == 4
    assert sum(isinstance(m, SystemMessage) for m in segunda) == 1
    assert isinstance(segunda[-1], HumanMessage)
    assert segunda[-1].content == "segunda pergunta"
    assert isinstance(segunda[-2], AIMessage)


def test_threads_distintos_isolam_estado():
    llm = FakeLLM()
    cp = MemorySaver()

    _executar("pergunta A", llm, cp, "thread_a")
    _executar("pergunta B", llm, cp, "thread_b")

    assert len(llm.invokes[0]) == 2
    assert isinstance(llm.invokes[0][0], SystemMessage)
    assert llm.invokes[0][-1].content == "pergunta A"
    assert len(llm.invokes[1]) == 2
    assert isinstance(llm.invokes[1][0], SystemMessage)
    assert llm.invokes[1][-1].content == "pergunta B"


def test_thread_id_none_mantem_comportamento_atual():
    llm = FakeLLM()
    historico = [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": "olá"},
    ]

    agente_ia.executar_agente(
        "nova pergunta",
        historico=historico,
        llm=llm,
        tools=ferramentas.FERRAMENTAS,
    )

    primeira = llm.invokes[0]
    assert isinstance(primeira[0], SystemMessage)
    assert agente_ia.PROMPT_AGENTE in primeira[0].content
    assert [m.content for m in primeira[1:-1]] == ["oi", "olá"]
    assert isinstance(primeira[-1], HumanMessage)
    assert primeira[-1].content == "nova pergunta"
