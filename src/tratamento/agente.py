import os

import questionary
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown

from src.tratamento.documentos_apoio import _FONTES_APOIO
from src.tratamento.geração import (
    SOBRE_YNUVASHA,
    _e_pergunta_sobre_si,
    gerar_resposta,
)
from src.tratamento.retrieval import (
    formatar_contexto,
    k_para_pergunta,
    recuperar_contexto_com_apoio,
)

NUM_PEDACOS = 5
NUM_TROCAS_MEMORIA = 4  # trocas (pergunta + resposta) mantidas no histórico da conversa

# Observabilidade por turno. Desligado por padrão; ative com
# YNUYASHA_LOG_TURNOS=true no .env para gravar cada turno em CSV.
LOGAR_TURNOS = os.getenv("YNUYASHA_LOG_TURNOS", "false").lower() == "true"

COMANDOS_SAIDA = {"sair", "quit", "exit"}
COMANDO_NOVA_CONVERSA = {"nova conversa", "novo"}

CONSOLE = Console()

ESTILO_PROMPT = Style(
    [
        ("qmark", "bold cyan"),
        ("question", "bold white"),
        ("answer", "bold cyan"),
        ("instruction", "dim white"),
    ]
)


def _registrar_turno(pergunta: str, resposta: str) -> None:
    """Grava o turno em CSV quando YNUYASHA_LOG_TURNOS=true."""
    if not LOGAR_TURNOS:
        return
    from src.tratamento.avaliacao import registrar_turno

    registrar_turno(
        pergunta,
        resposta,
        latencia_s=0.0,
        citou_fonte=("Fonte:" in resposta or "## Fontes" in resposta),
        apoio=any(nome in resposta for nome in _FONTES_APOIO),
        modo="rag",
    )


def responder(pergunta: str, historico=None) -> str:
    """Pipeline completo do agente: recuperação → geração (com memória multi-turno)."""
    if _e_pergunta_sobre_si(pergunta):
        resposta = gerar_resposta(pergunta, SOBRE_YNUVASHA, historico=historico)
        _registrar_turno(pergunta, resposta)
        return resposta
    k = k_para_pergunta(pergunta, NUM_PEDACOS)
    resultados = recuperar_contexto_com_apoio(pergunta, k=k, historico=historico)
    contexto = formatar_contexto(resultados)
    resposta = gerar_resposta(pergunta, contexto, historico=historico)
    _registrar_turno(pergunta, resposta)
    return resposta


def preparar_contexto(pergunta: str, k: int = NUM_PEDACOS, historico=None) -> str:
    """Recupera e formata o contexto RAG de uma pergunta (sem gerar resposta).

    Permite que a interface exiba os pedaços recuperados e reutilize o mesmo
    contexto na geração em streaming, evitando novas chamadas de embedding.
    """
    if _e_pergunta_sobre_si(pergunta):
        return SOBRE_YNUVASHA
    k = k_para_pergunta(pergunta, k)
    resultados = recuperar_contexto_com_apoio(pergunta, k=k, historico=historico)
    return formatar_contexto(resultados)


def _loop_cli() -> None:
    CONSOLE.rule("[bold magenta]Ynuyasha[/bold magenta] — Agente especializado em astronomia (RAG)")
    CONSOLE.print(
        "[dim]Digite 'sair' ou 'quit' para encerrar; 'nova conversa' reinicia a memória.[/dim]"
    )
    historico = []
    while True:
        try:
            pergunta = questionary.text(
                "Você:",
                qmark="»",
                style=ESTILO_PROMPT,
                instruction="Digite sua pergunta e pressione Enter",
            ).ask()
        except (EOFError, KeyboardInterrupt):
            CONSOLE.print("\n[italic]Até mais![/italic]")
            break
        if pergunta is None:
            CONSOLE.print("[italic]Até mais![/italic]")
            break
        pergunta = pergunta.strip()
        if not pergunta:
            continue
        if pergunta.lower() in COMANDOS_SAIDA:
            CONSOLE.print("[italic]Até mais![/italic]")
            break
        if pergunta.lower() in COMANDO_NOVA_CONVERSA:
            historico = []
            CONSOLE.print("[italic]Nova conversa iniciada! 🌌[/italic]")
            continue
        CONSOLE.print("\n[bold magenta]Ynuyasha:[/bold magenta]")
        try:
            resposta = responder(pergunta, historico=list(historico))
            CONSOLE.print(Markdown(resposta))
        except Exception as erro:
            CONSOLE.print(f"[bold red][ERRO][/bold red] Não foi possível responder: {erro}")
            resposta = f"(erro) {erro}"
        historico.append({"role": "user", "content": pergunta})
        historico.append({"role": "assistant", "content": resposta})
        historico = historico[-NUM_TROCAS_MEMORIA * 2:]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pergunta = " ".join(sys.argv[1:])
        CONSOLE.print(Markdown(responder(pergunta)))
    else:
        _loop_cli()
