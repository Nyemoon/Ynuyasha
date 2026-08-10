import csv
import os
import requests


def obter_eventos_transientes():
    print("Consultando SIMBAD (Supernovas, Púlsares e Eventos Extremos)...")
    url = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"

    tipos_eventos = ['QSO', 'Psr', 'SN*', 'BH?']
    linhas_totais = []

    for tipo in tipos_eventos:
        # Busca até 14 registros de cada tipo para garantir balanceamento (pois BH? tem no max 14)
        query = f"SELECT TOP 14 main_id, otype, ra, dec FROM basic WHERE otype = '{tipo}'"
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
            linhas_totais.extend(linhas)
        else:
            print(f"   -> Erro na consulta SIMBAD para {tipo}: {res.status_code}")

    if linhas_totais:
        try:
            caminho_script = os.path.abspath(__file__)
        except NameError:
            caminho_script = os.path.abspath(".")

        diretorio_destino = os.path.join(
            os.path.dirname(caminho_script), "..", "..", "data", "dataset"
        )
        os.makedirs(diretorio_destino, exist_ok=True)
        arquivo_csv = os.path.join(
            diretorio_destino, "eventos_transientes_extremos.csv"
        )

        with open(arquivo_csv, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "identificador_evento",
                "tipo_evento_astrofisico",
                "ascensao_reta_ra",
                "declinacao_dec",
                "fonte_dados",
                "status_validacao",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for linha_dado in linhas_totais:
                writer.writerow({
                    "identificador_evento": linha_dado[0] if linha_dado[0] else "N/A",
                    "tipo_evento_astrofisico": linha_dado[1] if linha_dado[1] else "N/A",
                    "ascensao_reta_ra": linha_dado[2] if linha_dado[2] is not None else "N/A",
                    "declinacao_dec": linha_dado[3] if linha_dado[3] is not None else "N/A",
                    "fonte_dados": "SIMBAD High Energy Database",
                    "status_validacao": "Validado",
                })
        print(
            f"   -> Sucesso! Arquivo '{arquivo_csv}' gerado com {len(linhas_totais)} registros balanceados.\n"
        )
    else:
        print("   -> Erro: Nenhum dado foi retornado nas consultas SIMBAD.\n")


if __name__ == "__main__":
    obter_eventos_transientes()