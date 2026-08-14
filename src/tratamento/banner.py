"""Geração do banner do Ynuyasha (cachorrinho + nome em figlet).

Compartilhado entre a interface de terminal (``main.py``) e a interface web
(``interface/app.py``) para que o mesmo banner apareça em ambas.
"""

import pyfiglet

FONTE_BANNER = "slant"

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


def gerar_linhas_banner() -> list[tuple[str, str]]:
    """Retorna as linhas do banner como pares ``(arte_cachorro, arte_nome)``.

    Ambas as partes já vêm alinhadas (mesma altura), prontas para serem
    coloridas de forma independente pelo terminal ou pela página web.
    """
    linhas = pyfiglet.figlet_format("Ynuyasha", font=FONTE_BANNER).rstrip("\n").split("\n")

    altura = max(len(linhas), len(ARTE_CACHORRO))
    arte_nome = [""] * (altura - len(linhas)) + list(linhas)
    cachorro = [_LINHA_VAZIA_CACHORRO] * (altura - len(ARTE_CACHORRO)) + ARTE_CACHORRO

    return list(zip(cachorro, arte_nome))
