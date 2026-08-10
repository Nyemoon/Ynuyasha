from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent  # sobe de testes/ pra raiz do projeto
DATASET_DIR = BASE_DIR / "data" / "dataset"

arquivos = [
    "asteroides_cometas_jpl.csv",
    "estrelas_e_objetos_simbad.csv",
    "estrelas_proximas_gaia.csv",
    "eventos_transientes_extremos.csv",
    "planetas_e_estrelas_rag.csv",
    "habitabilidade_exoplanetas.csv",
    "glossario_astronomico_conceitos.csv",
    "constelacoes_iau.csv",
]

for nome_arquivo in arquivos:
    print(f"{nome_arquivo}\n")
    df = pd.read_csv(DATASET_DIR / nome_arquivo)
    print(df.isna().sum())
    print("\n")