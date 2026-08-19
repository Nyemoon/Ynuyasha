"""Seleção de porta livre para a interface web (stdlib puro, sem Gradio)."""

import socket

LIMITE_SONDAS = 50


def procurar_porta_livre(inicio: int = 7860) -> int:
    """Retorna a primeira porta livre a partir de `inicio`.

    Sonda com `socket.bind` para não conflitar com instâncias já em execução;
    levanta OSError se nenhuma porta no intervalo estiver disponível.
    """
    for porta in range(inicio, inicio + LIMITE_SONDAS):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", porta))
            except OSError:
                continue
            return porta
    raise OSError(
        f"Nenhuma porta livre entre {inicio} e {inicio + LIMITE_SONDAS - 1}. "
        "Encerre processos Gradio antigos e tente novamente."
    )
