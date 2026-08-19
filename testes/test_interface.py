"""Testes da interface web (portas, banner, status e chat)."""

import pytest
import socket as socket_mod

from interface import app, portas


# ─── Seleção de porta livre ───────────────────────────────────────────────────


class _SocketFalso:
    """Context manager que simula `socket.socket` com portas ocupadas."""

    def __init__(self, ocupadas: set):
        self.ocupadas = ocupadas

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def bind(self, endereco):
        if endereco[1] in self.ocupadas:
            raise OSError("porta ocupada")


def test_procurar_porta_livre_pula_ocupadas(monkeypatch):
    ocupadas = {7860, 7861}
    monkeypatch.setattr(socket_mod, "socket", lambda *a, **k: _SocketFalso(ocupadas))
    assert portas.procurar_porta_livre(7860) == 7862


def test_procurar_porta_livre_sem_ocupadas(monkeypatch):
    monkeypatch.setattr(socket_mod, "socket", lambda *a, **k: _SocketFalso(set()))
    assert portas.procurar_porta_livre(7860) == 7860


def test_procurar_porta_livre_esgota_intervalo(monkeypatch):
    ocupadas = set(range(portas.LIMITE_SONDAS))
    monkeypatch.setattr(socket_mod, "socket", lambda *a, **k: _SocketFalso(ocupadas))
    with pytest.raises(OSError):
        portas.procurar_porta_livre(0)


# ─── Banner ───────────────────────────────────────────────────────────────────


def test_html_banner_inclui_cachorro_e_nome():
    html = app._html_banner()
    assert '<div id="app-banner">' in html
    assert "banner-cao" in html
    assert "banner-nome" in html
    assert "<pre>" in html


# ─── Status ───────────────────────────────────────────────────────────────────


def _status_fake() -> dict:
    return {
        "geracao": "Groq (openai/gpt-oss-120b)",
        "groq_configurada": True,
        "ollama_online": True,
        "documentos": 123,
        "rag_limiar": 0.65,
        "modelo_embedding": "nomic-embed-text",
        "fallback": "smollm2:360m",
        "agente": "RAG simples",
        "fingerprint_atual": "abc",
        "fingerprint_indexado": "abc",
        "datasets_sincronizados": True,
    }


def test_montar_status_html_renderiza_badges(monkeypatch):
    monkeypatch.setattr(app, "verificar_status", lambda forcar_ollama=False: _status_fake())
    html = app.montar_status_html()
    assert "badge-ok" in html
    assert "Online" in html
    assert "Sincronizada" in html
    assert "123 docs" in html
    assert "nomic-embed-text" in html


def test_montar_status_html_escapa_valores(monkeypatch):
    status = _status_fake()
    status["modelo_embedding"] = '<script>alert(1)</script>'
    status["ollama_online"] = False
    status["datasets_sincronizados"] = False
    monkeypatch.setattr(app, "verificar_status", lambda forcar_ollama=False: status)
    html = app.montar_status_html()
    assert "&lt;script&gt;" in html
    assert "<script>" not in html
    assert "badge-err" in html
    assert "badge-warn" in html


# ─── Chat ─────────────────────────────────────────────────────────────────────


def test_responder_chat_pergunta_vazia():
    saida = next(app.responder_chat("", []))
    assert len(saida) == 8
    assert saida[0] == []
    assert saida[2] == "Pronto"


def test_responder_chat_comando_saida():
    saida = next(app.responder_chat("sair", []))
    assert len(saida) == 8
    assert saida[0][-1]["role"] == "assistant"
    assert saida[2] == "Sessão encerrada"


def test_responder_chat_streaming(monkeypatch):
    monkeypatch.setattr(
        app, "preparar_contexto", lambda pergunta, k=5, historico=None: "[1] (relevância: 0.8, fonte: a.csv, linha: 1)"
    )
    monkeypatch.setattr(
        app, "gerar_resposta_stream", lambda pergunta, ctx, historico=None: iter(["Olá", " mundo"])
    )
    saidas = list(app.responder_chat("O que é um parsec?", []))
    final = saidas[-1]
    assert len(final) == 8
    assert final[0][-1]["role"] == "assistant"
    assert final[0][-1]["content"] == "Olá mundo"
    assert final[4] == ["O que é um parsec?", "Olá mundo"]
    assert final[5] == {"__type__": "update"}


def test_responder_chat_envia_historico_recente_ao_retrieval(monkeypatch):
    capturado = {}

    def fake_preparar_contexto(pergunta, k=5, historico=None):
        capturado["historico"] = list(historico or [])
        return "[1] (relevância: 0.8, fonte: a.csv, linha: 1)"

    monkeypatch.setattr(app, "preparar_contexto", fake_preparar_contexto)
    monkeypatch.setattr(
        app, "gerar_resposta_stream", lambda pergunta, ctx, historico=None: iter(["ok"])
    )
    historico_anterior = [
        {"role": "user", "content": "O que é um parsec?"},
        {"role": "assistant", "content": "Parsec é uma unidade de distância."},
    ]
    list(app.responder_chat("e quanto vale em anos-luz?", historico_anterior))
    assert capturado["historico"] == historico_anterior


def test_limpar_conversa_restaura_boas_vindas():
    saida = app.limpar_conversa()
    assert len(saida) == 8
    assert saida[0] == []
    assert saida[5] == {"__type__": "update"}


def test_normalizar_historico_converte_blocos_do_chatbot():
    historico_ui = [
        {"role": "user", "content": [{"text": "O que é um parsec?", "type": "text"}]},
        {"role": "assistant", "content": [{"text": "Parsec é...", "type": "text"}]},
        {"role": "user", "content": "texto simples"},
    ]
    limpo = app._normalizar_historico(historico_ui)
    assert limpo[0] == {"role": "user", "content": "O que é um parsec?"}
    assert limpo[1] == {"role": "assistant", "content": "Parsec é..."}
    assert limpo[2] == {"role": "user", "content": "texto simples"}
    assert all(isinstance(m["content"], str) for m in limpo)