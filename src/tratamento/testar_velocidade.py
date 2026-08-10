import time

from src.tratamento.embeddings import embeddings_model

TEXTO_TESTE = "planeta com água líquida em órbita de uma estrela anã vermelha"

print("Chamada 1 (pode incluir cold start — Ollama carregando o modelo)...")
inicio = time.time()
vetor = embeddings_model.embed_query(TEXTO_TESTE)
duracao_1 = time.time() - inicio
print(f"Levou {duracao_1:.2f} segundos")
print(f"Tamanho do vetor: {len(vetor)} dimensões")

print("\nChamada 2 (modelo já deveria estar carregado em memória)...")
inicio = time.time()
vetor = embeddings_model.embed_query(TEXTO_TESTE)
duracao_2 = time.time() - inicio
print(f"Levou {duracao_2:.2f} segundos")

print("\nChamada 3 (confirmando estabilidade)...")
inicio = time.time()
vetor = embeddings_model.embed_query(TEXTO_TESTE)
duracao_3 = time.time() - inicio
print(f"Levou {duracao_3:.2f} segundos")

# Estimativa usando a chamada 3 (mais representativa do custo real por
# chamada, já sem qualquer efeito de cold start das anteriores).
print(f"\nEstimativa para 190 pedaços (baseada na chamada 3): "
      f"~{duracao_3 * 190:.1f} segundos (~{duracao_3 * 190 / 60:.1f} minutos)")