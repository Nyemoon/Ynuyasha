import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pyfiglet
import questionary
from dotenv import load_dotenv
from prompt_toolkit.styles import Style
from questionary import Choice, Separator
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

FONTE_BANNER = "slant"
CAMINHO_ENV = Path(__file__).resolve().parent / ".env"
load_dotenv(CAMINHO_ENV)

CAMINHO_VECTORSTORE = Path(__file__).resolve().parent / "data" / "vectorstore" / "embeddings_store.json"
PLACEHOLDER_CHAVE = "sua_chave_aqui"

# ─── Tema de cores centralizado ────────────────────────────────────────────
TEMA = Theme(
    {
        "sucesso": "bold green",
        "erro": "bold red",
        "aviso": "yellow",
        "destaque": "bold cyan",
        "agente": "bold magenta",
        "fonte": "dim white",
        "dado": "cyan",
    }
)

CONSOLE = Console(theme=TEMA)

ESTILO_MENU = Style(
    [
        ("qmark", "bold cyan"),
        ("question", "bold white"),
        ("answer", "bold cyan"),
        ("pointer", "bold cyan"),
        ("highlighted", "bold white bg:grey"),
        ("separator", "dim magenta"),
        ("instruction", "dim white"),
    ]
)

SPINNERS_TEMATICOS = ["moon", "earth", "star", "dots"]


# ─── Banner ─────────────────────────────────────────────────────────────────

ARTE_CACHORRO_BRUTA = [
    "  /\\_/\\  ",
    " ( o.o ) ",
    " >  w  < ",
    "/|     |\\",
    "(_|   |_)",
    "  U   U  ",
]
_LARGURA_CACHORRO = max(len(linha) for linha in ARTE_CACHORRO_BRUTA)
ARTE_CACHORRO = [linha.ljust(_LARGURA_CACHORRO) for linha in ARTE_CACHORRO_BRUTA]
_LINHA_VAZIA_CACHORRO = " " * _LARGURA_CACHORRO


def mostrar_banner() -> None:
    """Exibe o banner (cachorrinho + nome Ynuyasha) centralizado no terminal."""
    linhas = pyfiglet.figlet_format("Ynuyasha", font=FONTE_BANNER).rstrip("\n").split("\n")
    largura_nome = max(len(linha) for linha in linhas)
    
    altura = max(len(linhas), len(ARTE_CACHORRO))
    arte_nome = [""] * (altura - len(linhas)) + list(linhas)
    cachorro = [_LINHA_VAZIA_CACHORRO] * (altura - len(ARTE_CACHORRO)) + ARTE_CACHORRO

    # Largura total do bloco combinado (cachorrinho + 2 espaços de respiro + texto figlet)
    largura_bloco = _LARGURA_CACHORRO + 2 + largura_nome
    
    # Calcula a margem esquerda com base na largura atual do terminal
    largura_terminal = CONSOLE.width
    margem_esquerda = max(0, (largura_terminal - largura_bloco) // 2)
    espacos_margem = " " * margem_esquerda

    for linha_nome, linha_cachorro in zip(arte_nome, cachorro):
        linha = Text()
        linha.append(espacos_margem)
        linha.append(linha_cachorro, style="magenta")
        linha.append("  ")
        linha.append(linha_nome, style="bold green")
        CONSOLE.print(linha)


# ─── Status do sistema ───────────────────────────────────────────────────────

def _groq_configurada() -> bool:
    chave = os.getenv("GROQ_API_KEY", "").strip()
    return bool(chave) and chave != PLACEHOLDER_CHAVE


def _modelo_fallback() -> str:
    return os.getenv("OLLAMA_FALLBACK_MODEL", "smollm2:360m")


_OLLAMA_URL = "http://localhost:11434/api/tags"
_OLLAMA_CACHE_TTL = 30.0
_ultima_verificacao_ollama = 0.0
_cache_ollama_online = False


def _ollama_online(forcar: bool = False) -> bool:
    global _ultima_verificacao_ollama, _cache_ollama_online
    agora = time.monotonic()
    if not forcar and (agora - _ultima_verificacao_ollama) < _OLLAMA_CACHE_TTL:
        return _cache_ollama_online
    try:
        with urllib.request.urlopen(_OLLAMA_URL, timeout=2) as resp:
            _cache_ollama_online = resp.status == 200
    except (urllib.error.URLError, OSError):
        _cache_ollama_online = False
    _ultima_verificacao_ollama = agora
    return _cache_ollama_online


def _contar_documentos() -> int | None:
    if not CAMINHO_VECTORSTORE.exists():
        return None
    try:
        with open(CAMINHO_VECTORSTORE, encoding="utf-8") as f:
            return len(json.load(f))
    except (OSError, json.JSONDecodeError):
        return None


def verificar_status(forcar_ollama: bool = False) -> dict:
    groq = _groq_configurada()
    return {
        "geracao": (
            f"Groq ({os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')})"
            if groq
            else f"Fallback Ollama ({_modelo_fallback()})"
        ),
        "groq_configurada": groq,
        "ollama_online": _ollama_online(forcar=forcar_ollama),
        "documentos": _contar_documentos(),
    }


def exibir_painel_status(status: dict) -> None:
    docs = status["documentos"]
    texto_docs = f"{docs} documento(s)" if docs is not None else "[aviso]não construída[/aviso]"

    tabela = Table(box=box.ROUNDED, title="[bold cyan]STATUS DO SISTEMA[/bold cyan]", title_justify="left", expand=True)
    tabela.add_column("Componente", style="bold magenta", width=18)
    tabela.add_column("Situação", style="white")
    
    tabela.add_row("Motor de Geração", status["geracao"])
    
    status_groq = "[sucesso]✔ Configurada[/sucesso]" if status["groq_configurada"] else "[erro]✖ Ausente[/erro]"
    tabela.add_row("Chave Groq API", status_groq)

    status_ollama = "[sucesso]✔ Online[/sucesso]" if status["ollama_online"] else "[erro]✖ Offline[/erro]"
    tabela.add_row("Servidor Ollama", status_ollama)

    tabela.add_row("Base Vetorial", texto_docs)
    
    CONSOLE.print(tabela)


# ─── Consultas de dados (com barra de progresso) ────────────────────────────

def executar_consultas() -> None:
    CONSOLE.clear()
    CONSOLE.rule("[destaque]🛰️  PIPELINE DE CONSULTAS DE DADOS[/destaque]")
    CONSOLE.print()

    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    pasta_consultas = os.path.join(diretorio_atual, "src", "consultas")
    scripts_para_ignorar = ["__init__.py"]

    if not os.path.exists(pasta_consultas):
        CONSOLE.print(f"[erro]✖ Erro:[/erro] O diretório de consultas não foi encontrado: {pasta_consultas}")
        _pausar()
        return

    scripts = [f for f in os.listdir(pasta_consultas) if f.endswith(".py") and f not in scripts_para_ignorar]

    if not scripts:
        CONSOLE.print("[aviso]⚠ Nenhum script de consulta encontrado na pasta.[/aviso]")
        _pausar()
        return

    CONSOLE.print(f"Foram encontrados [destaque]{len(scripts)}[/destaque] scripts para execução.\n")

    sucessos = 0
    falhas = 0
    resultados_detalhados = []

    with Progress(
        SpinnerColumn(spinner_name="earth"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, complete_style="cyan", finished_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=CONSOLE,
    ) as progress:
        tarefa = progress.add_task("Processando...", total=len(scripts))

        for script in sorted(scripts):
            progress.update(tarefa, description=f"Executando [cyan]{script}[/cyan]")
            caminho_script = os.path.join(pasta_consultas, script)

            try:
                resultado = subprocess.run(
                    [sys.executable, caminho_script],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                sucessos += 1
                resultados_detalhados.append((script, True, resultado.stdout.strip(), resultado.stderr.strip()))
            except subprocess.CalledProcessError as e:
                falhas += 1
                resultados_detalhados.append((script, False, (e.stdout or "").strip(), (e.stderr or "").strip()))

            progress.advance(tarefa)

    CONSOLE.print("\n")
    for script, ok, stdout, stderr in resultados_detalhados:
        status_txt = "[sucesso][OK][/sucesso]" if ok else "[erro][FALHOU][/erro]"
        CONSOLE.print(f"• [bold]{script}[/bold] {status_txt}")
        if stdout:
            CONSOLE.print(Panel(stdout, box=box.SIMPLE, style="dim"))
        if stderr:
            estilo = "aviso" if ok else "erro"
            CONSOLE.print(Panel(stderr, title="Mensagem", border_style=estilo, box=box.SIMPLE))

    CONSOLE.print()
    CONSOLE.print(
        Panel(
            f"[sucesso]✔ {sucessos} com sucesso[/sucesso]  |  [erro]✖ {falhas} com falhas[/erro]",
            title="[bold]Resumo do Pipeline[/bold]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    _pausar()


# ─── Vectorstore ──────────────────────────────────────────────────────────────

def reconstruir_vectorstore() -> None:
    CONSOLE.clear()
    CONSOLE.rule("[destaque]🌌 RECONSTRUÇÃO DA VECTORSTORE[/destaque]")
    CONSOLE.print()
    
    try:
        from src.tratamento.base_vetorial import criar_ou_carregar_vectorstore
        with CONSOLE.status("[agente]Gerando embeddings via Ollama...[/agente]", spinner="earth"):
            vectorstore = criar_ou_carregar_vectorstore(forcar_rebuild=True)
        CONSOLE.print(f"\n[sucesso]✔ Vectorstore reconstruída com sucesso![/sucesso] Total: [destaque]{len(vectorstore.store)}[/destaque] documentos.")
    except Exception as erro:
        CONSOLE.print(f"\n[erro]✖ Falha crítica ao construir a vectorstore:[/erro] {erro}")
    
    _pausar()


# ─── Agente ───────────────────────────────────────────────────────────────────

def iniciar_agente_interativo() -> None:
    from src.tratamento.agente import _loop_cli
    _loop_cli()


def _separar_fontes(texto: str) -> tuple[str, str | None]:
    padrao = re.compile(
        r"^[ \t]*#{1,6}[ \t]+fontes?[ \t]*:?[ \t]*$"
        r"|^[ \t]*\*\*fontes?\*\*[ \t]*:?[ \t]*$"
        r"|^[ \t]*fontes?[ \t]*:",
        re.IGNORECASE | re.MULTILINE,
    )
    correspondencia = padrao.search(texto)
    if correspondencia:
        inicio = correspondencia.start()
        corpo, fontes = texto[:inicio].strip(), texto[correspondencia.end():].strip()
        return corpo, fontes or None
    return texto, None


def fazer_pergunta_unica() -> None:
    CONSOLE.clear()
    CONSOLE.rule("[destaque]💬 CONSULTA RÁPIDA AO AGENTE[/destaque]")
    CONSOLE.print()

    from src.tratamento.agente import responder

    pergunta = questionary.text(
        "Digite sua pergunta:",
        qmark="❯",
        style=ESTILO_MENU,
        instruction="Pressione Enter para enviar",
    ).ask()
    
    if not pergunta or not pergunta.strip():
        CONSOLE.print("[aviso]Nenhuma pergunta informada.[/aviso]")
        _pausar()
        return
        
    pergunta = pergunta.strip()
    spinner_escolhido = random.choice(SPINNERS_TEMATICOS)
    
    CONSOLE.print()
    try:
        with CONSOLE.status("[agente]🔭 Consultando a base de conhecimento...[/agente]", spinner=spinner_escolhido):
            resposta = responder(pergunta)
    except Exception as erro:
        CONSOLE.print(f"\n[erro]✖ Não foi possível processar a resposta:[/erro] {erro}")
        _pausar()
        return

    corpo, fontes = _separar_fontes(resposta)

    CONSOLE.print()
    CONSOLE.print(
        Panel(
            Markdown(corpo),
            title="[agente]✦ Resposta do Ynuyasha[/agente]",
            border_style="magenta",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )

    if fontes:
        CONSOLE.print(
            Panel(
                Markdown(fontes),
                title="[fonte]📚 Fontes de Referência[/fonte]",
                border_style="dim",
                box=box.SIMPLE,
                padding=(0, 2),
            )
        )
    
    _pausar()


def exibir_status_detalhado() -> None:
    CONSOLE.clear()
    CONSOLE.rule("[destaque]⚙️ DIAGNÓSTICO DO SISTEMA[/destaque]")
    CONSOLE.print()
    status = verificar_status(forcar_ollama=True)
    exibir_painel_status(status)
    CONSOLE.print()
    _pausar()


def _pausar() -> None:
    CONSOLE.print()
    questionary.press_any_key_to_continue(
        message="Pressione qualquer tecla para retornar ao menu...",
        style=ESTILO_MENU
    ).ask()


# ─── Menu Principal ───────────────────────────────────────────────────────────

def exibir_menu(status: dict) -> str:
    CONSOLE.clear()
    mostrar_banner()
    CONSOLE.print()
    exibir_painel_status(status)
    CONSOLE.print()

    opcoes = [
        Separator(" 🛰️  DADOS & PIPELINE "),
        Choice("Executar consultas (baixar/atualizar datasets)", value="1"),
        Choice("(Re)construir vectorstore (Embeddings Ollama)", value="2"),
        Separator(" 🌌 AGENTE INTELIGENTE (RAG) "),
        Choice("Iniciar agente interativo (Conversa contínua)", value="3"),
        Choice("Fazer uma pergunta única (Consulta rápida)", value="4"),
        Choice("Executar fluxo completo (Consultas + Vectorstore + Agente)", value="5"),
        Separator(" ⚙️  SISTEMA "),
        Choice("Ver diagnóstico completo do sistema", value="6"),
        Choice("Sair da aplicação", value="0"),
    ]

    selecionado = questionary.select(
        "Selecione a operação desejada:",
        choices=opcoes,
        qmark="❯",
        pointer="▶",
        instruction="Use as setas (↑/↓) para navegar e Enter para confirmar",
        style=ESTILO_MENU,
    ).ask()
    return selecionado or "0"


def main() -> None:
    try:
        while True:
            status_atual = verificar_status()
            opcao = exibir_menu(status_atual)

            if opcao == "1":
                executar_consultas()
            elif opcao == "2":
                reconstruir_vectorstore()
            elif opcao == "3":
                iniciar_agente_interativo()
            elif opcao == "4":
                fazer_pergunta_unica()
            elif opcao == "5":
                executar_consultas()
                reconstruir_vectorstore()
                iniciar_agente_interativo()
            elif opcao == "6":
                exibir_status_detalhado()
            elif opcao == "0":
                CONSOLE.clear()
                CONSOLE.print(Panel("[bold]🌙 Encerrando o Ynuyasha. Até logo![/bold]", border_style="dim", box=box.ROUNDED))
                break
    except (KeyboardInterrupt, EOFError):
        CONSOLE.clear()
        CONSOLE.print(Panel("[bold]🌙 Sessão interrompida. Desligando Ynuyasha...[/bold]", border_style="dim", box=box.ROUNDED))


if __name__ == "__main__":
    main()