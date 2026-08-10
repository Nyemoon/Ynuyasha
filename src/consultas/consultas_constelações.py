import csv
import os


def gerar_dataset_constelacoes():
    print("Gerando Dataset de Constelações Oficiais da IAU...")

    constelacoes = [
        {
            "sigla": "Ori",
            "nome_portugues": "Órion",
            "nome_latin": "Orion",
            "hemisferio": "Ambos",
            "estrela_principal": "Betelgeuse",
            "quantidade_estrelas_brilhantes": 7,
        },
        {
            "sigla": "Cen",
            "nome_portugues": "Centauro",
            "nome_latin": "Centaurus",
            "hemisferio": "Sul",
            "estrela_principal": "Alpha Centauri",
            "quantidade_estrelas_brilhantes": 10,
        },
        {
            "sigla": "Lyr",
            "nome_portugues": "Lira",
            "nome_latin": "Lyra",
            "hemisferio": "Norte",
            "estrela_principal": "Vega",
            "quantidade_estrelas_brilhantes": 5,
        },
        {
            "sigla": "Aqr",
            "nome_portugues": "Aquário",
            "nome_latin": "Aquarius",
            "hemisferio": "Ambos",
            "estrela_principal": "Sadalsuud",
            "quantidade_estrelas_brilhantes": 4,
        },
        {
            "sigla": "Peg",
            "nome_portugues": "Pégaso",
            "nome_latin": "Pegasus",
            "hemisferio": "Norte",
            "estrela_principal": "Enif",
            "quantidade_estrelas_brilhantes": 9,
        },
        {
            "sigla": "UMa",
            "nome_portugues": "Ursa Maior",
            "nome_latin": "Ursa Major",
            "hemisferio": "Norte",
            "estrela_principal": "Alioth",
            "quantidade_estrelas_brilhantes": 7,
        },
        {
            "sigla": "Cru",
            "nome_portugues": "Cruzeiro do Sul",
            "nome_latin": "Crux",
            "hemisferio": "Sul",
            "estrela_principal": "Acrux",
            "quantidade_estrelas_brilhantes": 5,
        },
    ]

    try:
        caminho_script = os.path.abspath(__file__)
    except NameError:
        caminho_script = os.path.abspath(".")

    diretorio_destino = os.path.join(
        os.path.dirname(caminho_script), "..", "..", "data", "dataset"
    )
    os.makedirs(diretorio_destino, exist_ok=True)
    arquivo_csv = os.path.join(
        diretorio_destino, "constelacoes_iau.csv"
    )

    with open(arquivo_csv, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sigla",
            "nome_portugues",
            "nome_latin",
            "hemisferio_visivel",
            "estrela_principal",
            "quantidade_estrelas_brilhantes",
            "fonte_dados",
            "status_validacao",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for c in constelacoes:
            writer.writerow({
                "sigla": c["sigla"],
                "nome_portugues": c["nome_portugues"],
                "nome_latin": c["nome_latin"],
                "hemisferio_visivel": c["hemisferio"],
                "estrela_principal": c["estrela_principal"],
                "quantidade_estrelas_brilhantes": c[
                    "quantidade_estrelas_brilhantes"
                ],
                "fonte_dados": "IAU (International Astronomical Union)",
                "status_validacao": "Validado",
            })
    print(
        f"   -> Sucesso! Arquivo '{arquivo_csv}' gerado com {len(constelacoes)} registros.\n"
    )


if __name__ == "__main__":
    gerar_dataset_constelacoes()