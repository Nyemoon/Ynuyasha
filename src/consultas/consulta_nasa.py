import csv
import requests
import os

# 1. Lista unificada com os 31 planetas do seu dataset
planetas = [
    # Bloco 1
    "51 Peg b", "55 Cnc e", "Kepler-452 b", "Proxima Cen b", "TRAPPIST-1 b",
    # Bloco 2
    "GJ 581 c", "GJ 667 C c", "Kepler-1649 c", "Kepler-186 f", "Kepler-22 b",
    "Kepler-442 b", "Kepler-62 f", "TOI-700 d",
    # Bloco 3
    "55 Cnc f", "70 Vir b", "GJ 1214 b", "HD 189733 b", "HR 8799 b",
    "K2-18 b", "KELT-9 b", "Kepler-11 c", "Kepler-16 b", "WASP-12 b",
    # Bloco 4
    "HD 209458 b", "Kepler-10 b", "TRAPPIST-1 c", "TRAPPIST-1 d",
    "TRAPPIST-1 e", "TRAPPIST-1 f", "TRAPPIST-1 g", "TRAPPIST-1 h"
]

# Formata a lista para o formato SQL: 'Planeta 1','Planeta 2',...
lista_formatada = ",".join([f"'{p}'" for p in planetas])

# 2. Query ADQL em uma única linha (evita erros de parsing no servidor da NASA)
query = f"SELECT pl_name, hostname, disc_year, discoverymethod, pl_rade, pl_masse, pl_eqt, st_spectype, st_teff, st_rad, st_mass, sy_dist FROM pscomppars WHERE pl_name IN ({lista_formatada})"

# 3. URL base do serviço TAP da NASA
url_base = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Parâmetros enviados via POST
payload = {
    'query': query,
    'format': 'json'
}

print("Enviando requisição para o NASA Exoplanet Archive...")

# 4. Requisição POST
resposta = requests.post(url_base, data=payload)

if resposta.status_code == 200:
    dados = resposta.json()
    print(f"Sucesso! {len(dados)} registros retornados da NASA.\n")

    diretorio_destino = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "dataset")
    os.makedirs(diretorio_destino, exist_ok=True)
    arquivo_csv = os.path.join(diretorio_destino, "planetas_e_estrelas_rag.csv")

    colunas_mapa = {
        "pl_name": "nome_planeta",
        "hostname": "nome_estrela",
        "disc_year": "ano_descoberta",
        "discoverymethod": "metodo_descoberta",
        "pl_rade": "raio_terrestre",
        "pl_masse": "massa_terrestre",
        "pl_eqt": "temperatura_planeta_k",
        "st_spectype": "tipo_espectral_estrela",
        "st_teff": "temperatura_estrela_k",
        "st_rad": "raio_solar_estrela",
        "st_mass": "massa_solar_estrela",
        "sy_dist": "distancia_parsecs",
        "fonte_dados": "fonte_dados",
        "status_validacao": "status_validacao"
    }

    # 5. Gravando o arquivo CSV
    with open(arquivo_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(colunas_mapa.values()))
        writer.writeheader()

        for item in dados:
            linha = {
                "nome_planeta": item.get("pl_name", "N/A"),
                "nome_estrela": item.get("hostname", "N/A"),
                "ano_descoberta": item.get("disc_year", "N/A"),
                "metodo_descoberta": item.get("discoverymethod", "N/A"),
                "raio_terrestre": item.get("pl_rade") if item.get("pl_rade") is not None else "N/A",
                "massa_terrestre": item.get("pl_masse") if item.get("pl_masse") is not None else "N/A",
                "temperatura_planeta_k": item.get("pl_eqt") if item.get("pl_eqt") is not None else "N/A",
                "tipo_espectral_estrela": item.get("st_spectype") if item.get("st_spectype") is not None else "N/A",
                "temperatura_estrela_k": item.get("st_teff") if item.get("st_teff") is not None else "N/A",
                "raio_solar_estrela": item.get("st_rad") if item.get("st_rad") is not None else "N/A",
                "massa_solar_estrela": item.get("st_mass") if item.get("st_mass") is not None else "N/A",
                "distancia_parsecs": item.get("sy_dist") if item.get("sy_dist") is not None else "N/A",
                "fonte_dados": "NASA Exoplanet Archive (pscomppars)",
                "status_validacao": "Validado"
            }
            writer.writerow(linha)

    print(f"Arquivo '{arquivo_csv}' gerado com sucesso!")

else:
    print(f"Erro ao consultar a API. Código de status: {resposta.status_code}")
    print(resposta.text)


if __name__ == "__main__":
    pass