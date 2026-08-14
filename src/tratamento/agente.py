import os

import questionary
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown

from src.tratamento.geração import gerar_resposta
from src.tratamento.memoria import novo_thread_id
from src.tratamento.retrieval import formatar_contexto, recuperar_contexto

NUM_PEDACOS = 5
NUM_TROCAS_MEMORIA = 4  # trocas (pergunta + resposta) mantidas no histórico da conversa

# Observabilidade por turno. Desligado por padrão; ative com
# YNUVASHA_LOG_TURNOS=true no .env para gravar cada turno em CSV.
LOGAR_TURNOS = os.getenv("YNUVASHA_LOG_TURNOS", "false").lower() == "true"

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


def _registrar_turno(pergunta: str, resposta: str, metadados: dict | None = None) -> None:
    """Grava o turno em CSV quando YNUVASHA_LOG_TURNOS=true.

    Reaproveita os metadados calculados por agente_ia (modo, tool_calls,
    latencia_s, citou_fonte) para não duplicar lógica nem perder a latência real.
    No caminho RAG (sem Groq) os metadados vêm vazios, com defaults coerentes.
    """
    if not LOGAR_TURNOS:
        return
    metadados = metadados or {}
    from src.tratamento.avaliacao import registrar_turno

    registrar_turno(
        pergunta,
        resposta,
        ferramentas_chamadas=metadados.get("tool_calls", 0),
        latencia_s=metadados.get("latencia_s", 0.0),
        citou_fonte=metadados.get("citou_fonte", "Fonte:" in resposta),
        modo=metadados.get("modo", "rag"),
    )


def responder(pergunta: str, historico=None, thread_id=None) -> str:
    """Pipeline completo do agente: recuperação → geração (com memória multi-turno).

    Com chave Groq configurada, usa o agente ReAct com tool-calling
    (agente_ia.executar_agente), que persiste o estado quando thread_id é dado;
    caso contrário, mantém o fluxo RAG clássico.
    """
    from src.tratamento.agente_ia import executar_agente, groq_disponivel

    if groq_disponivel():
        metadados = {}

        def registrador(md: dict) -> None:
            metadados.update(md)

        resposta = executar_agente(
            pergunta,
            historico=historico,
            thread_id=thread_id,
            registrador=registrador,
        )
        _registrar_turno(pergunta, resposta, metadados)
        return resposta

    resultados = recuperar_contexto(pergunta, k=NUM_PEDACOS)
    contexto = formatar_contexto(resultados)
    resposta = gerar_resposta(pergunta, contexto, historico=historico)
    _registrar_turno(pergunta, resposta)
    return resposta


def preparar_contexto(pergunta: str, k: int = NUM_PEDACOS) -> str:
    """Recupera e formata o contexto RAG de uma pergunta (sem gerar resposta).

    Permite que a interface exiba os pedaços recuperados e reutilize o mesmo
    contexto na geração em streaming, evitando novas chamadas de embedding.
    """
    resultados = recuperar_contexto(pergunta, k=k)
    return formatar_contexto(resultados)


def _loop_cli() -> None:
    CONSOLE.rule("[bold magenta]Ynuyasha[/bold magenta] — Agente especializado em astronomia (RAG)")
    CONSOLE.print(
        "[dim]Digite 'sair' ou 'quit' para encerrar; 'nova conversa' reinicia a memória.[/dim]"
    )
    thread_id = novo_thread_id()
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
            thread_id = novo_thread_id()
            historico = []
            CONSOLE.print("[italic]Nova conversa iniciada! 🌌[/italic]")
            continue
        CONSOLE.print("\n[bold magenta]Ynuyasha:[/bold magenta]")
        try:
            resposta = responder(pergunta, historico=list(historico), thread_id=thread_id)
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
        CONSOLE.print(Markdown(responder(pergunta, thread_id=novo_thread_id())))
    else:
        _loop_cli()
