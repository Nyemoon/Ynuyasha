import argparse
import html
import sys
import threading
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

import gradio as gr  # noqa: E402

from src.tratamento.agente import (  # noqa: E402
    NUM_PEDACOS,
    NUM_TROCAS_MEMORIA,
    preparar_contexto,
    responder,
)
from src.tratamento.agente_ia import groq_disponivel  # noqa: E402
from src.tratamento.avaliacao import registrar_feedback  # noqa: E402
from src.tratamento.banner import gerar_linhas_banner  # noqa: E402
from src.tratamento.embeddings import aquecer_embeddings  # noqa: E402
from src.tratamento.geração import gerar_resposta_stream  # noqa: E402
from src.tratamento.memoria import novo_thread_id  # noqa: E402
from src.tratamento.status import verificar_status  # noqa: E402

# Pré-carrega o modelo de embeddings no Ollama em segundo plano para que a
# primeira pergunta do RAG não pague o custo de carga do modelo (~30s).
threading.Thread(target=aquecer_embeddings, daemon=True).start()

COMANDOS_SAIDA = {"sair", "quit", "exit", "adeus", "tchau"}

_thread_id = novo_thread_id()  # thread da sessão ativa (persistência do agente)

EXEMPLOS = [
    "O que é um parsec?",
    "Qual a temperatura de equilíbrio do TRAPPIST-1 e?",
    "Quais planetas estão potencialmente na zona habitável?",
    "Qual o método de descoberta do Kepler-452 b?",
    "Qual a estrela principal de Órion?",
    "Quais asteroides são potencialmente perigosos?",
    "Explique o método de trânsito.",
    "Qual o tipo de objeto M 31?",
]

CSS = """
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }
#banner { text-align: center; padding: 0.4rem 0 0.2rem 0; container-type: inline-size; }
#banner pre {
  display: inline-block;
  text-align: left;
  font-family: "SF Mono", "Cascadia Code", "JetBrains Mono", Consolas, monospace;
  font-size: min(0.85rem, 2.9cqw);
  line-height: 1.15;
  background: transparent;
  border: none;
  padding: 0;
  margin-bottom: 0;
  overflow: hidden;
}
#subtitulo { text-align: center; margin-top: -0.3rem; }
#status { text-align: center; }
"""

TEMA = gr.themes.Soft(primary_hue="violet", secondary_hue="indigo", neutral_hue="slate")


def _texto_sincronia(status: dict) -> str:
    sincronizado = status["datasets_sincronizados"]
    if sincronizado is True:
        return "🟢 Sincronizada"
    if sincronizado is False:
        return "🔴 Desatualizada"
    return "🟡 Sem referência"


def mostrar_banner_html() -> str:
    """Comprime o banner do terminal em HTML colorido (cachorro + nome)."""
    cor_cachorro = "color:#b45fe0;"
    cor_nome = "color:#2e9e4f;"
    arte = "\n".join(
        f'<span style="{cor_cachorro}">{html.escape(cachorro)}</span>'
        f'<span style="{cor_nome}">{html.escape(nome)}</span>'
        for cachorro, nome in gerar_linhas_banner()
    )
    return (
        '<div id="banner" aria-label="Ynuyasha — Agente RAG de Astronomia">'
        f"<pre>{arte}</pre></div>"
    )


def montar_status() -> str:
    """Monta a linha de status do sistema (motor, Ollama, base vetorial e datasets)."""
    status = verificar_status(forcar_ollama=True)
    documentos = status["documentos"]
    texto_documentos = f"{documentos} documentos" if documentos is not None else "não construída"
    ollama = "🟢 Online" if status["ollama_online"] else "🔴 Offline"
    return (
        f"**Motor de geração:** {status['geracao']} &nbsp;|&nbsp; "
        f"**Ollama:** {ollama} &nbsp;|&nbsp; **Base vetorial:** {texto_documentos} &nbsp;|&nbsp; "
        f"**Limiar RAG:** {status['rag_limiar']:.2f} &nbsp;|&nbsp; "
        f"**Embedding:** {status['modelo_embedding']} &nbsp;|&nbsp; "
        f"**Fallback:** {status['fallback']} &nbsp;|&nbsp; "
        f"**Agente:** {status['agente']} &nbsp;|&nbsp; "
        f"**Datasets:** {_texto_sincronia(status)}"
    )


def _registrar_mensagens(historico: list, pergunta: str, resposta: str) -> list:
    mensagens = list(historico)
    mensagens.append({"role": "user", "content": pergunta})
    mensagens.append({"role": "assistant", "content": resposta})
    return mensagens


def responder_chat(pergunta, historico):
    """Pipeline completo (retrieval → streaming) exibindo também o contexto RAG."""
    historico = list(historico or [])
    pergunta = (pergunta or "").strip()
    ultima = [None, None]  # [pergunta, resposta] para o feedback

    if not pergunta:
        yield historico, "", "", "", ultima
        return

    if pergunta.lower() in COMANDOS_SAIDA:
        resposta = "Até mais! 🌙 Volte sempre que quiser explorar o cosmos comigo."
        yield _registrar_mensagens(historico, pergunta, resposta), "", "", "", ultima
        return

    yield historico, "🔭 Buscando contexto na base de conhecimento...", "🔄 Processando...", "", ultima

    historico_recente = list(historico)[-NUM_TROCAS_MEMORIA * 2:]

    if groq_disponivel():
        # Modo agente ReAct (tool-calling via Groq): resposta rápida e direta.
        yield historico, "", "🔮 Agente ReAct consultando as ferramentas...", "", ultima
        try:
            resposta = responder(pergunta, historico=historico_recente, thread_id=_thread_id)
        except Exception as erro:
            resposta = f"⚠️ Não foi possível gerar a resposta: {erro}"
            ultima = [pergunta, resposta]
            yield _registrar_mensagens(historico, pergunta, resposta), "", "✅ Concluído", "", ultima
            return
        ultima = [pergunta, resposta]
        nota = "Resposta gerada pelo **agente ReAct** (ferramentas via Groq) — consulte as fontes citadas no texto."
        yield _registrar_mensagens(historico, pergunta, resposta), nota, "✅ Concluído", "", ultima
        return

    try:
        contexto = preparar_contexto(pergunta, k=NUM_PEDACOS)
    except Exception as erro:
        resposta = f"⚠️ Não foi possível recuperar o contexto da base de conhecimento: {erro}"
        ultima = [pergunta, resposta]
        yield _registrar_mensagens(historico, pergunta, resposta), "", "✅ Concluído", "", ultima
        return

    if not (contexto or "").strip():
        contexto_exibicao = "Nenhum trecho relevante encontrado na base de conhecimento."
    else:
        contexto_exibicao = contexto

    texto_parcial = ""
    try:
        for pedaco in gerar_resposta_stream(pergunta, contexto, historico=historico_recente):
            texto_parcial += pedaco
            yield _registrar_mensagens(historico, pergunta, texto_parcial), contexto_exibicao, "✨ Gerando resposta...", "", ultima
    except Exception as erro:
        resposta = f"⚠️ Não foi possível gerar a resposta: {erro}"
        ultima = [pergunta, resposta]
        yield _registrar_mensagens(historico, pergunta, resposta), contexto_exibicao, "✅ Concluído", "", ultima

    ultima = [pergunta, texto_parcial]
    yield _registrar_mensagens(historico, pergunta, texto_parcial), contexto_exibicao, "✅ Concluído", "", ultima


def _avaliar(ultima, nota, status):
    """Grava o feedback da última resposta e devolve uma mensagem de confirmação."""
    mensagem = status
    if ultima and ultima[0] and ultima[1]:
        registrar_feedback(ultima[0], ultima[1], "positivo" if nota == "sim" else "negativo")
        mensagem = "Obrigado! Seu feedback foi registrado."
    return mensagem


def _avaliar_positivo(ultima, status):
    return _avaliar(ultima, "sim", status)


def _avaliar_negativo(ultima, status):
    return _avaliar(ultima, "nao", status)


def limpar_conversa() -> tuple:
    """Limpa o chat, o contexto e o indicador de status (reinicia a memória)."""
    global _thread_id
    _thread_id = novo_thread_id()
    return [], "", "", "", [None, None]


def construir_demo() -> gr.Blocks:
    with gr.Blocks(title="Ynuyasha — Agente RAG de Astronomia") as demo:
        gr.Markdown(mostrar_banner_html())

        gr.Markdown(
            "_Agente RAG de Astronomia — pergunte sobre exoplanetas, estrelas, "
            "constelações, asteroides e muito mais. Respostas em português do Brasil, "
            "com fontes citadas._",
            elem_id="subtitulo",
        )

        with gr.Row():
            chatbot = gr.Chatbot(
                label="Conversa",
                height=480,
                render_markdown=True,
                placeholder="Faça uma pergunta sobre astronomia para começar!",
            )

        with gr.Row():
            botao_enviar = gr.Button("Enviar", variant="primary", scale=1)
            botao_positivo = gr.Button("Gostei", variant="secondary")
            botao_negativo = gr.Button("Não gostei", variant="secondary")
        ultima = gr.State([None, None])

        pergunta = gr.Textbox(
            label="Sua pergunta",
            placeholder="Ex.: O que é um parsec?",
        )

        gr.Examples(examples=EXEMPLOS, inputs=pergunta, label="Perguntas sugeridas")

        with gr.Accordion("Contexto recuperado (RAG)", open=False):
            contexto = gr.Textbox(
                label="Pedaços recuperados (fonte, linha e relevância)",
                lines=12,
                interactive=False,
                buttons=["copy"],
            )

        status = gr.Markdown("", elem_id="status")
        botao_limpar = gr.Button("Limpar conversa", variant="secondary", size="sm")

        botao_enviar.click(
            responder_chat,
            inputs=[pergunta, chatbot],
            outputs=[chatbot, contexto, status, pergunta, ultima],
        )
        pergunta.submit(
            responder_chat,
            inputs=[pergunta, chatbot],
            outputs=[chatbot, contexto, status, pergunta, ultima],
        )
        botao_limpar.click(limpar_conversa, outputs=[chatbot, contexto, status, pergunta, ultima])
        botao_positivo.click(_avaliar_positivo, inputs=[ultima, status], outputs=status)
        botao_negativo.click(_avaliar_negativo, inputs=[ultima, status], outputs=status)

        demo.load(montar_status, outputs=status)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Interface web Gradio do Ynuyasha")
    parser.add_argument("--port", type=int, default=7860, help="Porta do servidor (padrão: 7860)")
    parser.add_argument("--share", action="store_true", help="Gerar um link público temporário")
    args = parser.parse_args()

    demo = construir_demo()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=TEMA,
        css=CSS,
    )


if __name__ == "__main__":
    main()
