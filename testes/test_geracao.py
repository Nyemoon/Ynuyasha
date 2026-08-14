from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.tratamento import geração


class FakeLLM:
    def __init__(self, texto="resposta fake", levantar=False):
        self._texto = texto
        self._levantar = levantar

    def invoke(self, prompt):
        if self._levantar:
            raise RuntimeError("erro fake")
        return SimpleNamespace(content=self._texto)

    def stream(self, prompt):
        if self._levantar:
            raise RuntimeError("erro fake")
        yield SimpleNamespace(content=self._texto)


@pytest.fixture(autouse=True)
def reseta_estado_global():
    yield
    geração._llm = None
    geração._usando_groq = False


def test_historico_para_mensagens_converte_roles():
    mensagens = geração._historico_para_mensagens(
        [
            {"role": "user", "content": "oi"},
            {"role": "assistant", "content": "olá"},
        ]
    )
    assert isinstance(mensagens[0], HumanMessage)
    assert mensagens[0].content == "oi"
    assert isinstance(mensagens[1], AIMessage)
    assert mensagens[1].content == "olá"


def test_historico_para_mensagens_ignora_papeis_desconhecidos():
    mensagens = geração._historico_para_mensagens(
        [{"role": "system", "content": "ignorado"}]
    )
    assert mensagens == []


def test_historico_para_mensagens_none():
    assert geração._historico_para_mensagens(None) == []


def test_montar_prompt_estrutura_completa():
    mensagens = geração.montar_prompt(
        "O que é um parsec?",
        "contexto relevante",
        historico=[{"role": "user", "content": "oi"}],
    )
    assert isinstance(mensagens[0], SystemMessage)
    assert mensagens[0].content == geração.SISTEMA_ASTRONOMIA
    assert len(mensagens) == 3
    assert isinstance(mensagens[1], HumanMessage)
    assert isinstance(mensagens[2], HumanMessage)
    ultimo = mensagens[-1].content
    assert "CONTEXTO:" in ultimo
    assert "PERGUNTA DO USUÁRIO:" in ultimo
    assert "contexto relevante" in ultimo
    assert "O que é um parsec?" in ultimo


def test_montar_prompt_sem_historico():
    mensagens = geração.montar_prompt("pergunta", "contexto")
    assert len(mensagens) == 2
    assert isinstance(mensagens[1], HumanMessage)


def test_montar_prompt_contexto_vazio_usa_marcador():
    mensagens = geração.montar_prompt("pergunta", "   ")
    ultimo = mensagens[-1].content
    assert "Nenhum trecho relevante encontrado" in ultimo


def test_gerar_resposta_com_fake_llm(monkeypatch):
    monkeypatch.setattr(geração, "obter_llm", lambda: FakeLLM(texto="resposta fake"))
    resposta = geração.gerar_resposta("pergunta", "contexto")
    assert resposta == "resposta fake"


def test_gerar_resposta_contexto_vazio_nao_chama_llm(monkeypatch):
    def obter_llm_falha():
        raise AssertionError("LLM não deveria ser consultado sem contexto")

    monkeypatch.setattr(geração, "obter_llm", obter_llm_falha)
    resposta = geração.gerar_resposta("pergunta", "   ")
    assert resposta == geração.MENSAGEM_FORA_DA_BASE


def test_gerar_resposta_contexto_vazio_recusa_sem_invocar_llm(monkeypatch):
    chamadas = []

    def obter_llm_contador():
        chamadas.append(1)
        return FakeLLM(texto="resposta fake")

    monkeypatch.setattr(geração, "obter_llm", obter_llm_contador)
    resposta = geração.gerar_resposta("pergunta", "")
    assert resposta == geração.MENSAGEM_FORA_DA_BASE
    assert chamadas == []


def test_gerar_resposta_usa_fallback_apos_erro_groq(monkeypatch):
    falha = FakeLLM(levantar=True)
    monkeypatch.setattr(geração, "obter_llm", lambda: falha)
    monkeypatch.setattr(
        geração, "_criar_llm_fallback", lambda: FakeLLM(texto="resposta local")
    )
    geração._usando_groq = True

    resposta = geração.gerar_resposta("pergunta", "contexto")
    assert resposta == "resposta local"
    assert geração._usando_groq is False


def test_gerar_resposta_relanca_erro_sem_fallback(monkeypatch):
    falha = FakeLLM(levantar=True)
    monkeypatch.setattr(geração, "obter_llm", lambda: falha)
    geração._usando_groq = False

    with pytest.raises(RuntimeError):
        geração.gerar_resposta("pergunta", "contexto")


def test_gerar_resposta_stream_com_fake_llm(monkeypatch):
    monkeypatch.setattr(geração, "obter_llm", lambda: FakeLLM(texto="resposta fake"))
    texto = "".join(geração.gerar_resposta_stream("pergunta", "contexto"))
    assert texto == "resposta fake"


def test_gerar_resposta_stream_contexto_vazio_nao_chama_llm(monkeypatch):
    def obter_llm_falha():
        raise AssertionError("LLM não deveria ser consultado sem contexto")

    monkeypatch.setattr(geração, "obter_llm", obter_llm_falha)
    texto = "".join(geração.gerar_resposta_stream("pergunta", ""))
    assert texto == geração.MENSAGEM_FORA_DA_BASE


def test_gerar_resposta_stream_usa_fallback(monkeypatch):
    falha = FakeLLM(levantar=True)
    monkeypatch.setattr(geração, "obter_llm", lambda: falha)
    monkeypatch.setattr(
        geração, "_criar_llm_fallback", lambda: FakeLLM(texto="resposta local")
    )
    geração._usando_groq = True

    texto = "".join(geração.gerar_resposta_stream("pergunta", "contexto"))
    assert texto == "resposta local"
    assert geração._usando_groq is False
