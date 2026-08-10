import csv
import requests
import os

def obter_objetos_simbad():
    print("Consultando SIMBAD (Estrelas Notáveis e Nebulosas)...")
    url = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
    
    objetos = [
        'Betelgeuse', 'Proxima Centauri', 'Sirius', 'Vega', 'Aldebaran', 
        'M 42', 'M 31', 'M 1', 'M 57', 'M 45'
    ]
    lista_str = ", ".join([f"'{o}'" for o in objetos])
    
    # MUDANÇA: antes a query filtrava direto em basic.main_id, que só bate
    # com o identificador PRINCIPAL de cada objeto (geralmente uma designação
    # de catálogo, não o nome popular). Agora usamos JOIN com a tabela
    # ident, que guarda todos os apelidos conhecidos de cada objeto — isso
    # é o que permite "Betelgeuse" (apelido) encontrar o registro cujo
    # main_id de verdade é outra coisa (ex: "* alf Ori").
    query = f"""
        SELECT basic.main_id, basic.otype, basic.ra, basic.dec, basic.sp_type, basic.plx_value
        FROM basic
        JOIN ident ON basic.oid = ident.oidref
        WHERE ident.id IN ({lista_str})
    """
    
    payload = {
        'request': 'doQuery',
        'lang': 'adql',
        'format': 'json',
        'query': query
    }
    
    res = requests.post(url, data=payload)
    
    if res.status_code == 200:
        dados_json = res.json()
        linhas = dados_json.get("data", [])

        diretorio_destino = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "dataset")
        os.makedirs(diretorio_destino, exist_ok=True)
        arquivo_csv = os.path.join(diretorio_destino, "estrelas_e_objetos_simbad.csv")
        
        with open(arquivo_csv, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "identificador_principal", "tipo_objeto", "ascensao_reta_ra", 
                "declinacao_dec", "tipo_espectral", "paralaxe_mas", 
                "fonte_dados", "status_validacao"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for linha_dado in linhas:
                # Estrutura do retorno: [main_id, otype_long, ra, dec, sp_type, plx_value]
                writer.writerow({
                    "identificador_principal": linha_dado[0] if linha_dado[0] else "N/A",
                    "tipo_objeto": linha_dado[1] if linha_dado[1] else "N/A",
                    "ascensao_reta_ra": linha_dado[2] if linha_dado[2] is not None else "N/A",
                    "declinacao_dec": linha_dado[3] if linha_dado[3] is not None else "N/A",
                    "tipo_espectral": linha_dado[4] if linha_dado[4] else "N/A",
                    "paralaxe_mas": linha_dado[5] if linha_dado[5] is not None else "N/A",
                    "fonte_dados": "SIMBAD / CDS Strasbourg",
                    "status_validacao": "Validado"
                })
        print(f"   -> Sucesso! Arquivo '{arquivo_csv}' gerado com {len(linhas)} registros.\n")
    else:
        print(f"   -> Erro na consulta SIMBAD: {res.status_code}\n")


if __name__ == "__main__":
    obter_objetos_simbad()