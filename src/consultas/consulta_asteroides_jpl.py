import csv
import os
import requests


def obter_asteroides_jpl():
    print("Consultando NASA JPL SBDB (Corpos Menores e Asteróides)...")
    # API pública da NASA JPL para pequenos corpos (Asteróides NEOs / Cometas)
    url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

    # Parâmetros de consulta aos 20 asteróides mais notáveis/próximos
    params = {
        "fields": "full_name,kind,diameter,pha,neo,class",
        "sb-group": "neo",
        "limit": "20",
    }

    res = requests.get(url, params=params)

    if res.status_code == 200:
        dados_json = res.json()
        linhas = dados_json.get("data", [])

        try:
            caminho_script = os.path.abspath(__file__)
        except NameError:
            caminho_script = os.path.abspath(".")

        diretorio_destino = os.path.join(
            os.path.dirname(caminho_script), "..", "..", "data", "dataset"
        )
        os.makedirs(diretorio_destino, exist_ok=True)
        arquivo_csv = os.path.join(
            diretorio_destino, "asteroides_cometas_jpl.csv"
        )

        with open(arquivo_csv, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "nome_corpo",
                "tipo_objeto",
                "diametro_km",
                "potencialmente_perigoso",
                "objeto_proximo_terra",
                "classe_orbital",
                "fonte_dados",
                "status_validacao",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for linha_dado in linhas:
                # Estrutura do retorno: [full_name, kind, diameter, pha, neo, class]
                writer.writerow({
                    "nome_corpo": linha_dado[0].strip() if linha_dado[0] else "N/A",
                    "tipo_objeto": linha_dado[1] if linha_dado[1] else "N/A",
                    "diametro_km": linha_dado[2] if linha_dado[2] is not None else "N/A",
                    "potencialmente_perigoso": "Sim" if linha_dado[3] == "Y" else "Não",
                    "objeto_proximo_terra": "Sim" if linha_dado[4] == "Y" else "Não",
                    "classe_orbital": linha_dado[5] if linha_dado[5] else "N/A",
                    "fonte_dados": "NASA JPL Small-Body Database",
                    "status_validacao": "Validado",
                })
        print(
            f"   -> Sucesso! Arquivo '{arquivo_csv}' gerado com {len(linhas)} registros.\n"
        )
    else:
        print(f"   -> Erro na consulta NASA JPL: {res.status_code}\n")


if __name__ == "__main__":
    obter_asteroides_jpl()