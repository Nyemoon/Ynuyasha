import csv
import os
import requests


def obter_estrelas_gaia():
    print("Consultando ESA Gaia Archive (Estrelas Próximas)...")
    url = "https://gea.esac.esa.int/tap-server/tap/sync"

    # Busca as 20 estrelas mais próximas mapeadas pelo Gaia DR3 (paralaxe > 100 mas)
    query = """
        SELECT TOP 20 source_id, ra, dec, phot_g_mean_mag, parallax 
        FROM gaiadr3.gaia_source 
        WHERE parallax > 100 
        ORDER BY parallax DESC
    """

    payload = {
        "request": "doQuery",
        "lang": "adql",
        "format": "json",
        "query": query,
    }

    res = requests.post(url, data=payload)

    if res.status_code == 200:
        dados_json = res.json()
        linhas = dados_json.get("data", [])

        # Obter o diretório atual do script com fallback seguro
        try:
            caminho_script = os.path.abspath(__file__)
        except NameError:
            caminho_script = os.path.abspath(".")

        diretorio_destino = os.path.join(
            os.path.dirname(caminho_script), "..", "..", "data", "dataset"
        )
        os.makedirs(diretorio_destino, exist_ok=True)
        arquivo_csv = os.path.join(
            diretorio_destino, "estrelas_proximas_gaia.csv"
        )

        with open(arquivo_csv, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "id_fonte_gaia",
                "ascensao_reta_ra",
                "declinacao_dec",
                "magnitude_g",
                "paralaxe_mas",
                "fonte_dados",
                "status_validacao",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for linha_dado in linhas:
                # Estrutura do retorno: [source_id, ra, dec, phot_g_mean_mag, parallax]
                writer.writerow({
                    "id_fonte_gaia": str(linha_dado[0]) if linha_dado[0] else "N/A",
                    "ascensao_reta_ra": linha_dado[1] if linha_dado[1] is not None else "N/A",
                    "declinacao_dec": linha_dado[2] if linha_dado[2] is not None else "N/A",
                    "magnitude_g": linha_dado[3] if linha_dado[3] is not None else "N/A",
                    "paralaxe_mas": linha_dado[4] if linha_dado[4] is not None else "N/A",
                    "fonte_dados": "ESA Gaia DR3",
                    "status_validacao": "Validado",
                })
        print(
            f"   -> Sucesso! Arquivo '{arquivo_csv}' gerado com {len(linhas)} registros.\n"
        )
    else:
        print(f"   -> Erro na consulta ESA Gaia: {res.status_code}\n")


if __name__ == "__main__":
    obter_estrelas_gaia()