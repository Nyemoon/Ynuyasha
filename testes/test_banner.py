import pyfiglet

from src.tratamento.banner import ARTE_CACHORRO, _LARGURA_CACHORRO, gerar_linhas_banner


def test_banner_linhas_formadas():
    linhas = gerar_linhas_banner()
    assert linhas
    for cachorro, _nome in linhas:
        assert len(cachorro) == _LARGURA_CACHORRO


def test_banner_altura_coincide_com_arte():
    linhas_figlet = pyfiglet.figlet_format("Ynuyasha", font="slant").rstrip("\n").split("\n")
    altura = max(len(linhas_figlet), len(ARTE_CACHORRO))
    assert len(gerar_linhas_banner()) == altura


def test_banner_nome_igual_ao_figlet():
    linhas = [nome for _cachorro, nome in gerar_linhas_banner()]
    linhas_figlet = pyfiglet.figlet_format("Ynuyasha", font="slant").rstrip("\n").split("\n")
    altura = max(len(linhas_figlet), len(ARTE_CACHORRO))
    figlet_padded = [""] * (altura - len(linhas_figlet)) + linhas_figlet
    assert linhas == figlet_padded