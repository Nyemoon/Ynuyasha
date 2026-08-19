# ruff: noqa: E402
import argparse
import html
import sys
import threading
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

import gradio as gr  # noqa: E402

from interface.portas import procurar_porta_livre  # noqa: E402
from src.tratamento.agente import (  # noqa: E402
    NUM_PEDACOS,
    NUM_TROCAS_MEMORIA,
    preparar_contexto,
)
from src.tratamento.avaliacao import registrar_feedback  # noqa: E402
from src.tratamento.banner import gerar_linhas_banner  # noqa: E402
from src.tratamento.embeddings import aquecer_embeddings  # noqa: E402
from src.tratamento.geração import gerar_resposta_stream  # noqa: E402
from src.tratamento.status import verificar_status  # noqa: E402

COMANDOS_SAIDA = {"sair", "quit", "exit", "adeus", "tchau"}

EXEMPLOS = [
    "O que é um parsec?",
    "Explique o método de trânsito.",
    "Qual a temperatura de equilíbrio do TRAPPIST-1 e?",
    "Quais asteroides são potencialmente perigosos?",
]

CSS = """
:root {
    --bg-deep: #08060e;
    --bg-space: #151026;
    --text-strong: #f8fafc;
    --text-body: #e2e8f0;
    --text-faint: #cbd5e1;
    --text-muted: #94a3b8;
    --magenta-600: #c026d3;
    --magenta-500: #d946ef;
    --magenta-400: #e879f9;
    --magenta-300: #f0abfc;
    --magenta-200: #f5d0fe;
    --magenta-100: #fae8ff;
    --cyan-400: #22d3ee;
    --cyan-300: #67e8f9;
    --cyan-200: #a5f3fc;
    --cyan-100: #cffafe;
    --ok-bg: rgba(74, 222, 128, 0.15);
    --ok-color: #4ade80;
    --ok-border: rgba(74, 222, 128, 0.28);
    --err-bg: rgba(248, 113, 113, 0.15);
    --err-color: #f87171;
    --err-border: rgba(248, 113, 113, 0.28);
    --warn-bg: rgba(251, 191, 36, 0.15);
    --warn-color: #fbbf24;
    --warn-border: rgba(251, 191, 36, 0.28);
    --dim-bg: rgba(226, 232, 240, 0.09);
    --dim-color: #cbd5e1;
    --dim-border: rgba(226, 232, 240, 0.18);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
    --radius-input: 24px;
    --radius-pill: 999px;
    --shadow-input: 0 8px 32px rgba(0, 0, 0, 0.35);
    --shadow-magenta: 0 4px 18px rgba(217, 70, 239, 0.25);
    --shadow-green: 0 4px 16px rgba(74, 222, 128, 0.22);
    --shadow-red: 0 4px 16px rgba(248, 113, 113, 0.22);
    --transition: all 0.2s ease;
    --border-white: rgba(255, 255, 255, 0.08);
    --header-pad: 0.65rem;
}

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    background: linear-gradient(135deg, var(--bg-deep) 0%, var(--bg-space) 100%) !important;
    color: var(--text-body) !important;
    min-height: 100vh;
}

.gr-box, .gr-form, .gr-panel {
    background: transparent !important;
    border: none !important;
}

#app-shell {
    max-width: 850px !important;
    width: 100%;
    margin: 0 auto !important;
    padding: 0 0.75rem !important;
}

#top-header {
    position: sticky;
    top: 0;
    z-index: 30;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    padding: var(--header-pad) 0;
    border-bottom: 1px solid var(--border-white);
    background: linear-gradient(135deg, rgba(8, 6, 14, 0.94) 0%, rgba(21, 16, 38, 0.94) 100%);
    backdrop-filter: blur(14px);
}
#app-brand h1 {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-strong);
    letter-spacing: -0.02em;
    margin: 0;
}
#app-brand p {
    font-size: 0.78rem;
    color: var(--cyan-300);
    margin: 0.1rem 0 0 0;
}

#chat-scroll {
    padding: 0.75rem 0 !important;
}

#app-banner {
    text-align: center;
    margin-bottom: 0.5rem !important;
}
#app-banner pre {
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
    font-size: 0.7rem;
    line-height: 1.05;
    display: inline-block;
    margin: 0 auto;
    text-align: left;
    background: transparent !important;
    border: none !important;
}
.banner-cao {
    color: var(--magenta-400);
    font-weight: 700;
    text-shadow: 0 0 14px rgba(217, 70, 239, 0.5);
}
.banner-nome {
    color: var(--cyan-300);
    font-weight: 700;
    text-shadow: 0 0 12px rgba(103, 232, 249, 0.4);
}

#welcome {
    text-align: center;
    padding: 0.75rem 0 1.25rem !important;
}
#hero h2 {
    font-size: 1.4rem;
    font-weight: 650;
    letter-spacing: -0.02em;
    color: var(--text-strong);
    margin-bottom: 0.35rem !important;
}
#hero p {
    color: var(--text-faint);
    font-size: 0.88rem;
    line-height: 1.5;
    max-width: 680px;
    margin: 0 auto 1.25rem auto !important;
}

#cards-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.5rem;
    width: 100%;
}
button.topic-card {
    background: rgba(232, 121, 249, 0.08) !important;
    border: 1px solid rgba(232, 121, 249, 0.32) !important;
    border-radius: var(--radius-md) !important;
    color: var(--magenta-200) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 0.7rem 1rem !important;
    backdrop-filter: blur(10px) !important;
    transition: var(--transition) !important;
    text-align: center;
}
button.topic-card:hover {
    border-color: var(--magenta-300) !important;
    background: rgba(232, 121, 249, 0.16) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 22px rgba(232, 121, 249, 0.35) !important;
}

#status-panel {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 0.75rem;
    padding: 0.5rem 0;
}
.status-item {
    background: rgba(103, 232, 249, 0.06);
    border: 1px solid rgba(103, 232, 249, 0.2);
    backdrop-filter: blur(10px);
    border-radius: var(--radius-sm);
    padding: 0.5rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}
.status-label {
    font-size: 0.65rem;
    color: var(--cyan-300);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.status-val {
    font-size: 0.85rem;
    color: var(--text-strong);
    font-weight: 500;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.15rem 0.55rem;
    border-radius: var(--radius-pill);
    white-space: nowrap;
    width: fit-content;
}
.badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.badge-ok { background: var(--ok-bg); color: var(--ok-color); border: 1px solid var(--ok-border); }
.badge-err { background: var(--err-bg); color: var(--err-color); border: 1px solid var(--err-border); }
.badge-warn { background: var(--warn-bg); color: var(--warn-color); border: 1px solid var(--warn-border); }
.badge-dim { background: var(--dim-bg); color: var(--dim-color); border: 1px solid var(--dim-border); }

.gr-accordion {
    background: rgba(217, 70, 239, 0.05) !important;
    border: 1px solid rgba(232, 121, 249, 0.22) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--magenta-200) !important;
    margin-bottom: 0.5rem;
}

#rag-chunks-box,
#rag-chunks-box textarea {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(232, 121, 249, 0.18) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-body) !important;
    font-size: 0.85rem !important;
}

#chatbot-main {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
.bubble-wrap {
    background: transparent !important;
}
.panel.bot-row {
    background: rgba(217, 70, 239, 0.10) !important;
    border: 1px solid rgba(232, 121, 249, 0.25) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.6rem 1rem !important;
    color: var(--text-body) !important;
    font-size: 0.95rem;
    line-height: 1.6;
    backdrop-filter: blur(10px) !important;
}
.panel.bot-row .message {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    color: var(--text-body) !important;
}
.panel.user-row {
    background: linear-gradient(135deg, rgba(217, 70, 239, 0.10) 0%, rgba(232, 121, 249, 0.16) 100%) !important;
    border: 1px solid rgba(232, 121, 249, 0.24) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: var(--radius-xl) var(--radius-xl) 4px var(--radius-xl) !important;
    padding: 0.6rem 1.1rem !important;
    color: var(--text-strong) !important;
    font-size: 0.95rem;
    box-shadow: var(--shadow-magenta) !important;
}
.panel.bot-row .avatar-container { display: none !important; }

#status-indicator {
    display: none;
    align-items: center;
    justify-content: center;
    gap: 0.55rem;
    width: fit-content;
    margin: 0 auto 0.5rem auto;
    padding: 0.5rem 1.1rem;
    border-radius: var(--radius-pill);
    background: rgba(217, 70, 239, 0.12);
    border: 1px solid rgba(232, 121, 249, 0.35);
    color: var(--magenta-200);
    font-size: 0.85rem;
    font-weight: 500;
    backdrop-filter: blur(10px);
}
#status-indicator.active { display: inline-flex; }
#status-indicator.ok {
    background: var(--ok-bg) !important;
    border-color: var(--ok-border) !important;
    color: var(--ok-color) !important;
}
#status-indicator.ok .spinner { display: none; }
.spinner {
    width: 14px;
    height: 14px;
    border: 2px solid rgba(232, 121, 249, 0.25);
    border-top-color: var(--magenta-400);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
}
@keyframes spin { to { transform: rotate(360deg); } }

#bottom-bar {
    position: sticky;
    bottom: 0;
    z-index: 30;
    padding-top: 0.5rem;
    padding-bottom: 0.75rem;
    background: linear-gradient(180deg, rgba(21, 16, 38, 0) 0%, rgba(21, 16, 38, 0.94) 25%, rgba(21, 16, 38, 0.97) 100%);
    backdrop-filter: blur(10px);
}
#bottom-actions {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    justify-content: flex-start;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}

#input-wrapper {
    background: rgba(15, 14, 34, 0.6);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(232, 121, 249, 0.22);
    border-radius: var(--radius-input);
    padding: 0.4rem 0.5rem 0.4rem 1.25rem;
    box-shadow: var(--shadow-input);
    display: flex;
    align-items: center;
    transition: var(--transition);
}
#input-wrapper:focus-within {
    border-color: var(--cyan-300);
    box-shadow: 0 0 22px rgba(103, 232, 249, 0.28);
}
#input-wrapper textarea {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    font-size: 0.95rem !important;
    color: var(--text-strong) !important;
}
#input-wrapper textarea::placeholder {
    color: var(--text-muted) !important;
}

button.primary {
    background: linear-gradient(180deg, #f3e8ff 0%, #d8b4fe 40%, #c084fc 100%) !important;
    border: none !important;
    color: #3b0764 !important;
    border-radius: var(--radius-lg) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    position: relative;
    overflow: hidden;
    box-shadow: 0 3px 14px rgba(192, 132, 252, 0.45) !important;
    transition: var(--transition) !important;
}
button.primary::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, rgba(255, 255, 255, 0.45) 0%, rgba(255, 255, 255, 0) 45%) !important;
    border-radius: inherit;
    pointer-events: none;
}
button.primary:hover {
    filter: brightness(1.06) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(192, 132, 252, 0.55) !important;
}
button.primary:focus-visible {
    outline: 2px solid var(--cyan-300) !important;
    outline-offset: 2px;
}

button.secondary {
    background: rgba(226, 232, 240, 0.06) !important;
    border: 1px solid rgba(226, 232, 240, 0.16) !important;
    color: var(--text-faint) !important;
    border-radius: var(--radius-pill) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    backdrop-filter: blur(8px);
    box-shadow: none !important;
    transition: var(--transition) !important;
}
button.secondary:hover {
    background: rgba(226, 232, 240, 0.12) !important;
    border-color: rgba(255, 255, 255, 0.35) !important;
    color: var(--text-strong) !important;
    transform: translateY(-1px);
}
button.secondary:focus-visible {
    outline: 2px solid var(--cyan-300) !important;
    outline-offset: 2px;
}

#gostei-btn {
    background: var(--ok-bg) !important;
    border: 1px solid var(--ok-border) !important;
    color: var(--ok-color) !important;
    box-shadow: none !important;
}
#gostei-btn:hover {
    background: rgba(74, 222, 128, 0.22) !important;
    border-color: var(--ok-color) !important;
    color: #d1fae5 !important;
    transform: translateY(-1px);
    box-shadow: var(--shadow-green) !important;
}

#nao-gostei-btn {
    background: var(--err-bg) !important;
    border: 1px solid var(--err-border) !important;
    color: var(--err-color) !important;
    box-shadow: none !important;
}
#nao-gostei-btn:hover {
    background: rgba(248, 113, 113, 0.22) !important;
    border-color: var(--err-color) !important;
    color: #fee2e2 !important;
    transform: translateY(-1px);
    box-shadow: var(--shadow-red) !important;
}

#limpar-btn {
    background: rgba(34, 211, 238, 0.08) !important;
    border: 1px solid rgba(103, 232, 249, 0.25) !important;
    color: var(--cyan-200) !important;
    box-shadow: none !important;
}
#limpar-btn:hover {
    background: rgba(34, 211, 238, 0.18) !important;
    border-color: var(--cyan-300) !important;
    color: var(--cyan-100) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(34, 211, 238, 0.28) !important;
}

#copiar-btn {
    background: linear-gradient(180deg, #f3e8ff 0%, #d8b4fe 40%, #c084fc 100%) !important;
    border: none !important;
    color: #3b0764 !important;
    position: relative;
    overflow: hidden;
    box-shadow: 0 3px 14px rgba(192, 132, 252, 0.4) !important;
}
#copiar-btn::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, rgba(255, 255, 255, 0.45) 0%, rgba(255, 255, 255, 0) 45%) !important;
    border-radius: inherit;
    pointer-events: none;
}
#copiar-btn:hover {
    filter: brightness(1.06) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(192, 132, 252, 0.55) !important;
}
#copiar-btn:focus-visible {
    outline: 2px solid var(--cyan-300) !important;
    outline-offset: 2px;
}

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--magenta-500), var(--cyan-400));
    border-radius: var(--radius-pill);
    border: 2px solid var(--bg-deep);
}
::selection { background: rgba(34, 211, 238, 0.35); color: var(--text-strong); }

@media (max-width: 600px) {
    #app-shell { padding: 0 0.5rem; }
    #top-header { padding: 0.5rem 0; }
    #app-brand p { display: none; }
    #cards-row { grid-template-columns: 1fr; }
}

#chat-scroll {
    flex-grow: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}

#chatbot-main {
    flex-grow: 1 !important;
    height: auto !important;
}

#chatbot-main .wrap {
    height: 100% !important;
}
/* Melhora a tipografia e deixa o texto das respostas mais claro e legível */
#chatbot-main, #chatbot-main .message, #chatbot-main p, #chatbot-main li, #chatbot-main span {
    color: #f1f5f9 !important; /* Um tom de branco bem mais claro e suave */
    font-size: 0.96rem !important;
    line-height: 1.65 !important;
    letter-spacing: -0.01em !important;
}

/* Destaca títulos (h1, h2, h3) gerados pelo markdown */
#chatbot-main h1, #chatbot-main h2, #chatbot-main h3 {
    color: var(--cyan-200) !important;
    font-weight: 650 !important;
    margin-top: 0.8rem !important;
    margin-bottom: 0.4rem !important;
}

/* Estiliza listas para terem um respiro melhor */
#chatbot-main ul, #chatbot-main ol {
    padding-left: 1.2rem !important;
    margin: 0.5rem 0 !important;
}

#chatbot-main li {
    margin-bottom: 0.35rem !important;
}

/* Deixa negritos mais vivos e destacados */
#chatbot-main strong {
    color: var(--magenta-200) !important;
    font-weight: 600 !important;
}

/* Estiliza linhas horizontais (hr) se houverem no texto */
#chatbot-main hr {
    border-color: rgba(232, 121, 249, 0.2) !important;
    margin: 1rem 0 !important;
}
"""

TEMA = gr.themes.Base(
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

def _esc(valor) -> str:
    return html.escape(str(valor), quote=True)


def _html_banner() -> str:
    linhas = gerar_linhas_banner()
    corpo = "\n".join(
        f'<span class="banner-cao">{_esc(cao)}</span>'
        f'<span class="banner-nome">{_esc(nome)}</span>'
        for cao, nome in linhas
    )
    return f'<div id="app-banner"><pre>{corpo}</pre></div>'


def _html_status(texto: str, ok: bool = False) -> str:
    if not texto:
        return ""
    classe = "ok" if ok else "active"
    return (
        f'<div id="status-indicator" class="{classe}">'
        '<span class="spinner"></span>'
        f"<span>{_esc(texto)}</span>"
        "</div>"
    )


def _html_hero() -> str:
    return (
        '<div id="hero">'
        "<h2>O que você quer aprender hoje?</h2>"
        "<p>Sou o Ynuyasha, um agente de IA baseado em RAG (Retrieval-Augmented Generation) "
        "especializado em astronomia. Respondo em português do Brasil, com fontes citadas e "
        "sempre com base na minha base de conhecimento: exoplanetas, estrelas, constelações, "
        "asteroides, cometas, zona habitável, glossário astronômico, nebulosas, quasares, "
        "púlsares, supernovas e diversos eventos astrofísicos.</p>"
        "</div>"
    )


def _texto_sincronia(status: dict) -> str:
    sincronizado = status["datasets_sincronizados"]
    if sincronizado is True:
        return "Sincronizada"
    if sincronizado is False:
        return "Desatualizada"
    return "Sem referência"


def montar_status_html() -> str:
    status = verificar_status(forcar_ollama=True)
    doc_count = status["documentos"]
    docs = f"{doc_count} docs" if doc_count is not None else "Vazio"
    ollama_txt = "Online" if status["ollama_online"] else "Offline"
    ollama_cls = "badge-ok" if status["ollama_online"] else "badge-err"
    sinc_txt = _texto_sincronia(status)
    sinc_cls = {
        "Sincronizada": "badge-ok",
        "Desatualizada": "badge-warn",
        "Sem referência": "badge-dim",
    }[sinc_txt]

    return f"""
<div id="status-panel">
  <div class="status-item"><span class="status-label">Motor</span><span class="status-val">{_esc(status['geracao'])}</span></div>
  <div class="status-item"><span class="status-label">Ollama</span><span class="badge {ollama_cls}">{ollama_txt}</span></div>
  <div class="status-item"><span class="status-label">Base Vetorial</span><span class="status-val">{_esc(docs)}</span></div>
  <div class="status-item"><span class="status-label">Datasets</span><span class="badge {sinc_cls}">{sinc_txt}</span></div>
  <div class="status-item"><span class="status-label">Limiar de relevância</span><span class="status-val">{status['rag_limiar']:.2f}</span></div>
  <div class="status-item"><span class="status-label">Embedding</span><span class="status-val">{_esc(status['modelo_embedding'])}</span></div>
</div>
"""


def _registrar_mensagens(historico: list, pergunta: str, resposta: str) -> list:
    mensagens = list(historico or [])
    mensagens.append({"role": "user", "content": pergunta})
    mensagens.append({"role": "assistant", "content": resposta})
    return mensagens


def _normalizar_historico(historico) -> list:
    limpo = []
    for msg in historico or []:
        if not isinstance(msg, dict):
            continue
        conteudo = msg.get("content")
        if isinstance(conteudo, list):
            texto = "".join(
                bloco.get("text", "") if isinstance(bloco, dict) else str(bloco)
                for bloco in conteudo
            )
        else:
            texto = str(conteudo or "")
        limpo.append({"role": msg.get("role", ""), "content": texto})
    return limpo


def responder_chat(pergunta, historico):
    historico = list(historico or [])
    pergunta = (pergunta or "").strip()
    ultima = [None, None]

    if not pergunta:
        yield historico, "", "Pronto", "", ultima, gr.update(), "", gr.update()
        return

    if pergunta.lower() in COMANDOS_SAIDA:
        resposta = "Sessão encerrada. Até que enfim."
        yield _registrar_mensagens(historico, pergunta, resposta), "", "Sessão encerrada", "", ultima, gr.update(), "", gr.update()
        return

    yield historico, "Buscando contexto...", "Processando...", "", ultima, gr.update(), _html_status("Processando..."), gr.update()

    historico_recente = _normalizar_historico(historico)[-NUM_TROCAS_MEMORIA * 2 :]

    try:
        contexto = preparar_contexto(pergunta, k=NUM_PEDACOS, historico=historico_recente)
    except Exception as erro:
        resposta = f"Falha na recuperação de contexto. Tente novamente: {erro}"
        ultima = [pergunta, resposta]
        yield _registrar_mensagens(historico, pergunta, resposta), "", "Erro", "", ultima, gr.update(), "", gr.update()
        return

    contexto_exibicao = contexto if (contexto or "").strip() else "Nenhum trecho de alta relevância encontrado. Terei que deduzir com base no que sei."

    texto_parcial = ""
    try:
        for pedaco in gerar_resposta_stream(pergunta, contexto, historico=historico_recente):
            texto_parcial += pedaco
            yield _registrar_mensagens(historico, pergunta, texto_parcial), contexto_exibicao, "Gerando...", "", ultima, gr.update(), _html_status("Gerando respostas..."), gr.update()
    except Exception as erro:
        resposta = f"Erro na geração. Minha paciência falhou (ou o servidor): {erro}"
        ultima = [pergunta, resposta]
        yield _registrar_mensagens(historico, pergunta, texto_parcial), contexto_exibicao, "Erro", "", ultima, gr.update(), "", gr.update()

    ultima = [pergunta, texto_parcial]
    yield _registrar_mensagens(historico, pergunta, texto_parcial), contexto_exibicao, "Pronto", "", ultima, gr.update(), _html_status("Concluído", ok=True), gr.update()


def _avaliar(ultima, nota, status):
    if ultima and ultima[0] and ultima[1]:
        registrar_feedback(ultima[0], ultima[1], "positivo" if nota == "sim" else "negativo")
        return "Feedback registrado."
    return status


def limpar_conversa():
    return [], "", "Pronto", "", [None, None], gr.update(), "", gr.update()


def _fazer_perguntar(exemplo):
    def _perguntar(historico):
        yield from responder_chat(exemplo, historico)

    return _perguntar


def construir_demo() -> gr.Blocks:
    input_texto = gr.Textbox(
        show_label=False,
        placeholder="Fique a vontade para perguntar. Estou esperando...",
        container=False,
        scale=9,
        lines=1,
        max_lines=5,
        render=False,
    )

    with gr.Blocks(title="Ynuyasha") as demo:

        with gr.Column(elem_id="app-shell"):

            with gr.Row(elem_id="top-header"):
                gr.HTML(
                    "<div id='app-brand'>"
                    "<h1>Ynuyasha</h1>"
                    "<p>Agente RAG de Astronomia</p>"
                    "</div>"
                )

            with gr.Column(elem_id="chat-scroll"):

                with gr.Column(elem_id="welcome"):
                    gr.HTML(_html_banner())
                    gr.HTML(_html_hero(), elem_id="hero")
                    with gr.Row(elem_id="cards-row"):
                        cards = [
                            gr.Button(exemplo, elem_classes=["topic-card"])
                            for exemplo in EXEMPLOS
                        ]

                with gr.Accordion("Telemetria e Base de Conhecimento", open=False):
                    status_ui = gr.HTML("Carregando telemetria...", elem_id="status-panel")
                    gr.Markdown("<br>**Trechos Recuperados (RAG)**")
                    contexto_ui = gr.Textbox(
                        show_label=False,
                        lines=6,
                        interactive=False,
                        container=False,
                        elem_id="rag-chunks-box",
                    )

                chatbot = gr.Chatbot(
                    elem_id="chatbot-main",
                    show_label=False,
                    render_markdown=True,
                    height="auto",
                    buttons=["copy"],
                )

                status_indicador = gr.HTML("", elem_id="status-indicator")

            ultima_interacao = gr.State([None, None])
            copia_estado = gr.State("")

            with gr.Column(elem_id="bottom-bar"):
                with gr.Row(elem_id="bottom-actions"):
                    btn_copiar = gr.Button("Copiar resposta", variant="secondary", size="sm", elem_id="copiar-btn")
                    btn_up = gr.Button("Gostei", variant="secondary", size="sm", elem_id="gostei-btn")
                    btn_down = gr.Button("Não gostei", variant="secondary", size="sm", elem_id="nao-gostei-btn")
                    btn_limpar = gr.Button("Nova Conversa", variant="secondary", size="sm", elem_id="limpar-btn")

                with gr.Row(elem_id="input-wrapper"):
                    input_texto.render()
                    btn_enviar = gr.Button("Enviar", variant="primary", scale=1, min_width=100)

            status_texto = gr.State("Pronto")

        saidas_chat = [chatbot, contexto_ui, status_texto, input_texto, ultima_interacao, gr.State(), status_indicador, gr.State()]

        input_texto.submit(
            responder_chat,
            inputs=[input_texto, chatbot],
            outputs=saidas_chat,
        )

        btn_enviar.click(
            responder_chat,
            inputs=[input_texto, chatbot],
            outputs=saidas_chat,
        )

        for card, exemplo in zip(cards, EXEMPLOS):
            card.click(
                _fazer_perguntar(exemplo),
                inputs=[chatbot],
                outputs=saidas_chat,
            )

        btn_limpar.click(
            limpar_conversa,
            outputs=saidas_chat,
        )

        btn_up.click(lambda u, s: _avaliar(u, "sim", s), inputs=[ultima_interacao, status_texto], outputs=status_texto)
        btn_down.click(lambda u, s: _avaliar(u, "nao", s), inputs=[ultima_interacao, status_texto], outputs=status_texto)

        btn_copiar.click(
            lambda ultima: (ultima or [None, None])[1] or "",
            inputs=[ultima_interacao],
            outputs=[copia_estado],
            js="(texto) => { navigator.clipboard.writeText(texto || ''); return texto; }",
        )

        demo.load(montar_status_html, outputs=status_ui)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    porta = procurar_porta_livre(args.port)

    threading.Thread(target=aquecer_embeddings, daemon=True).start()

    demo = construir_demo()
    demo.queue()

    print(f"Ynuyasha — Interface web em http://127.0.0.1:{porta}")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=porta,
        share=args.share,
        footer_links=[],
        theme=TEMA,
        css=CSS,
    )


if __name__ == "__main__":
    main()