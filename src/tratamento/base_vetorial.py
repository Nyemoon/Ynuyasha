import json
import os
from langchain_core.load import dumpd
from langchain_core.vectorstores import InMemoryVectorStore
from pathlib import Path
from src.tratamento.loading import obter_pedacos
from src.tratamento.embeddings import embeddings_model

CAMINHO_VECTORSTORE = Path(__file__).resolve().parents[2] / "data" / "vectorstore" / "embeddings_store.json"
BATCH_SIZE = 10  # número de requisições ao Ollama, como temos 190 pedaços, vão ser embeddings em blocos de 10.
CHECKPOINT_INTERVAL = 5  # salva a vectorstore em disco a cada N lotes (N * BATCH_SIZE documentos)


def _dump_json_compacto(valor: object, nivel: int = 0) -> str:
    """Serializa JSON mantendo dicionários legíveis em múltiplas linhas, mas
    listas (vetores) em linha única, separadas por vírgula."""
    pre = "  " * nivel
    if isinstance(valor, dict):
        if not valor:
            return "{}"
        linhas = []
        for chave, item in valor.items():
            if isinstance(item, dict):
                linhas.append(f"{pre}  {json.dumps(chave)}: {_dump_json_compacto(item, nivel + 1)}")
            else:
                linhas.append(f"{pre}  {json.dumps(chave)}: {json.dumps(item)}")
        return "{\n" + ",\n".join(linhas) + f"\n{pre}}}"
    return json.dumps(valor)


def _persistir(vectorstore: InMemoryVectorStore) -> None:
    """Salva a vectorstore em disco de forma atômica: grava em .tmp e renomeia."""
    CAMINHO_VECTORSTORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CAMINHO_VECTORSTORE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(_dump_json_compacto(dumpd(vectorstore.store)))
    os.replace(tmp, CAMINHO_VECTORSTORE)


def _carregar_se_completa(total_esperado: int) -> InMemoryVectorStore | None:
    """
    Tenta recarregar a vectorstore existente do disco. Só a considera válida
    se o número de documentos salvos bater com o total esperado de pedaços —
    um arquivo existente mas incompleto (ex: processo anterior interrompido
    no meio) é tratado como inválido, não como sucesso.

    Retorna a vectorstore se completa, ou None se não existir/estiver incompleta.
    """
    if not CAMINHO_VECTORSTORE.exists():
        return None

    vectorstore = InMemoryVectorStore.load(str(CAMINHO_VECTORSTORE), embedding=embeddings_model)
    total_salvo = len(vectorstore.store)

    if total_salvo == total_esperado:
        print(f"Recarregando vectorstore existente ({total_salvo}/{total_esperado} documentos, completa).")
        return vectorstore

    print(
        f"Vectorstore existente está incompleta ({total_salvo}/{total_esperado} documentos) "
        f"— provavelmente de uma execução anterior interrompida. Recriando do zero."
    )
    return None


def criar_ou_carregar_vectorstore(forcar_rebuild: bool = False) -> InMemoryVectorStore:
    """Cria a vectorstore a partir dos pedacos se o arquivo nao existir ou
    estiver incompleto; caso contrario recarrega sem recalcular os embeddings.

    Se forcar_rebuild for True, ignora o arquivo existente e recalcula tudo.
    """
    pedacos = obter_pedacos()
    total = len(pedacos)

    vectorstore_existente = None if forcar_rebuild else _carregar_se_completa(total)
    if vectorstore_existente is not None:
        return vectorstore_existente

    print("Criando nova vectorstore...")
    vectorstore = InMemoryVectorStore(embedding=embeddings_model)

    try:
        for i in range(0, total, BATCH_SIZE):
            batch = pedacos[i:i + BATCH_SIZE]
            vectorstore.add_documents(batch)
            lote = i // BATCH_SIZE + 1
            print(f"[{i + len(batch)}/{total}] embedados")

            # Dump intra-loop com checkpoint: persiste o progresso a cada
            # CHECKPOINT_INTERVAL lotes (ou no último lote), usando escrita
            # atômica (.tmp + os.replace) para evitar arquivos corrompidos.
            if lote % CHECKPOINT_INTERVAL == 0 or i + len(batch) >= total:
                _persistir(vectorstore)
                print(f"[checkpoint] vectorstore salva em {CAMINHO_VECTORSTORE}")
    except Exception:
        # Garante que o progresso parcial fique salvo em disco mesmo se o
        # Ollama falhar no meio — mas note que _carregar_se_completa() vai
        # detectar esse arquivo como incompleto na próxima execução e
        # recriar do zero, então nada de errado é assumido silenciosamente.
        _persistir(vectorstore)
        raise

    print(f"Vectorstore salva em {CAMINHO_VECTORSTORE}")
    return vectorstore


if __name__ == "__main__":
    vs = criar_ou_carregar_vectorstore()
    print(f"Vectorstore pronta com {len(vs.store)} documentos.")