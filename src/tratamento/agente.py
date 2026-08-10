import questionary
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown

from src.tratamento.geração import gerar_resposta
from src.tratamento.retrieval import recuperar_contexto, formatar_contexto

NUM_PEDACOS = 5

CONSOLE = Console()

ESTILO_PROMPT = Style(
    [
        ("qmark", "bold cyan"),
        ("question", "bold white"),
        ("answer", "bold cyan"),
        ("instruction", "dim white"),
    ]
)


def responder(pergunta: str) -> str:
    """Pipeline completo do agente: recuperação → geração."""
    resultados = recuperar_contexto(pergunta, k=NUM_PEDACOS)
    contexto = formatar_contexto(resultados)
    return gerar_resposta(pergunta, contexto)


def _loop_cli() -> None:
    CONSOLE.rule("[bold magenta]Ynuyasha[/bold magenta] — Agente especializado em astronomia (RAG)")
    CONSOLE.print("[dim]Digite 'sair' ou 'quit' para encerrar.[/dim]")
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
        if pergunta.lower() in {"sair", "quit", "exit"}:
            CONSOLE.print("[italic]Até mais![/italic]")
            break
        CONSOLE.print("\n[bold magenta]Ynuyasha:[/bold magenta]")
        try:
            CONSOLE.print(Markdown(responder(pergunta)))
        except Exception as erro:
            CONSOLE.print(f"[bold red][ERRO][/bold red] Não foi possível responder: {erro}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pergunta = " ".join(sys.argv[1:])
        CONSOLE.print(Markdown(responder(pergunta)))
    else:
        _loop_cli()
