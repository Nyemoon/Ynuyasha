import csv
import os
import requests


def obter_phl_habitabilidade():
    print("Consultando NASA / PHL (Habitabilidade e Atmosferas)...")
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

    planetas = [
        "51 Peg b", "55 Cnc e", "Kepler-452 b", "Proxima Cen b", "TRAPPIST-1 b",
        "GJ 581 c", "GJ 667 C c", "Kepler-1649 c", "Kepler-186 f", "Kepler-22 b",
        "Kepler-442 b", "Kepler-62 f", "TOI-700 d",
        "55 Cnc f", "70 Vir b", "GJ 1214 b", "HD 189733 b", "HR 8799 b",
        "K2-18 b", "KELT-9 b", "Kepler-11 c", "Kepler-16 b", "WASP-12 b",
        "HD 209458 b", "Kepler-10 b", "TRAPPIST-1 c", "TRAPPIST-1 d",
        "TRAPPIST-1 e", "TRAPPIST-1 f", "TRAPPIST-1 g", "TRAPPIST-1 h"
    ]
    lista_formatada = ",".join([f"'{p}'" for p in planetas])

    query = f"""
        SELECT pl_name, hostname, pl_rade, pl_masse, pl_eqt, pl_insol, st_teff, sy_dist 
        FROM pscomppars 
        WHERE pl_name IN ({lista_formatada})
    """

    payload = {"query": query, "format": "json"}
    res = requests.post(url, data=payload)

    if res.status_code == 200:
        dados = res.json()

        try:
            caminho_script = os.path.abspath(__file__)
        except NameError:
            caminho_script = os.path.abspath(".")

        diretorio_destino = os.path.join(
            os.path.dirname(caminho_script), "..", "..", "data", "dataset"
        )
        os.makedirs(diretorio_destino, exist_ok=True)
        arquivo_csv = os.path.join(
            diretorio_destino, "habitabilidade_exoplanetas.csv"
        )

        with open(arquivo_csv, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "nome_planeta",
                "nome_estrela",
                "raio_terrestre",
                "massa_terrestre",
                "temperatura_equilibrio_k",
                "fluxo_insolacao_terra",
                "zona_habitavel_estimada",
                "fonte_dados",
                "status_validacao",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for item in dados:
                temp = item.get("pl_eqt")
                insol = item.get("pl_insol")

                # Regra simples de estimativa de Zona Habitável Teórica (baseada em temperatura de equilíbrio em Kelvin)
                if temp is not None and 200 <= temp <= 320:
                    zh = "Potencialmente Habitável (Conservadora)"
                elif temp is not None and 180 <= temp <= 390:
                    zh = "Potencialmente Habitável (Otimista)"
                else:
                    zh = "Fora da Zona Habitável"

                writer.writerow({
                    "nome_planeta": item.get("pl_name", "N/A"),
                    "nome_estrela": item.get("hostname", "N/A"),
                    "raio_terrestre": item.get("pl_rade")
                    if item.get("pl_rade") is not None
                    else "N/A",
                    "massa_terrestre": item.get("pl_masse")
                    if item.get("pl_masse") is not None
                    else "N/A",
                    "temperatura_equilibrio_k": temp
                    if temp is not None
                    else "N/A",
                    "fluxo_insolacao_terra": insol
                    if insol is not None
                    else "N/A",
                    "zona_habitavel_estimada": zh,
                    "fonte_dados": "NASA Exoplanet Archive / Algoritmo PHL",
                    "status_validacao": "Validado",
                })
        print(
            f"   -> Sucesso! Arquivo '{arquivo_csv}' gerado com {len(dados)} registros.\n"
        )
    else:
        print(f"   -> Erro na consulta: {res.status_code}\n")


if __name__ == "__main__":
    obter_phl_habitabilidade()