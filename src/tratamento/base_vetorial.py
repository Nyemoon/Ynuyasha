import json
import os
from pathlib import Path

from filelock import FileLock
from langchain_core.load import dumpd
from langchain_core.vectorstores import InMemoryVectorStore

from src.tratamento.embeddings import embeddings_model
from src.tratamento.loading import calcular_fingerprint_datasets, obter_pedacos

CAMINHO_VECTORSTORE = Path(__file__).resolve().parents[2] / "data" / "vectorstore" / "embeddings_store.json"
CAMINHO_FINGERPRINT = CAMINHO_VECTORSTORE.parent / "fingerprint_datasets.txt"
CAMINHO_LOCK = CAMINHO_VECTORSTORE.parent / "rebuild.lock"
BATCH_SIZE = 10  # número de requisições ao Ollama, como temos 190 pedaços, vão ser embeddings em blocos de 10.
CHECKPOINT_INTERVAL = 5  # salva a vectorstore em disco a cada N lotes (N * BATCH_SIZE documentos)
REBUILD_TIMEOUT = 3600  # segundos máximos que outra execução pode estar reconstruindo


def salvar_fingerprint() -> None:
    """Grava o hash dos datasets que originaram o índice atual em disco."""
    CAMINHO_FINGERPRINT.parent.mkdir(parents=True, exist_ok=True)
    CAMINHO_FINGERPRINT.write_text(calcular_fingerprint_datasets(), encoding="utf-8")


def ler_fingerprint() -> str | None:
    """Lê o hash gravado na última reconstrução; None se nunca gravado."""
    if not CAMINHO_FINGERPRINT.exists():
        return None
    try:
        return CAMINHO_FINGERPRINT.read_text(encoding="utf-8").strip()
    except OSError:
        return None


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
        # Só evita reconstrução concorrente se já houver um índice completo.
        return vectorstore_existente

    # Lock de reconstrução: impede que duas execuções (ex.: o Gradio bootando
    # enquanto um rebuild em background está em andamento) embebam os mesmos
    # pedaços em paralelo — o que sobrecarrega o Ollama e corrompe o arquivo.
    # Se outro processo está reconstruindo, espera até ele terminar (até o
    # timeout) e então carrega o resultado completo.
    lock = FileLock(str(CAMINHO_LOCK))
    lock_obtido = False
    try:
        lock.acquire(timeout=REBUILD_TIMEOUT)
        lock_obtido = True
    except TimeoutError:
        print("Rebuild já em andamento por outro processo e não concluiu")
        print(f"dentro de {REBUILD_TIMEOUT}s; prosseguindo com o índice atual no disco.")
    try:
        # Outra execução pode ter completado o rebuild enquanto esperávamos o lock.
        vectorstore_existente = None if forcar_rebuild else _carregar_se_completa(total)
        if vectorstore_existente is not None:
            return vectorstore_existente

        if not lock_obtido:
            # Não podemos reconstruir em paralelo com outro processo: carrega o
            # que estiver no disco (mesmo incompleto) para não bloquear o RAG.
            print("Aguardando o rebuild em andamento; usando o índice parcial disponível.")
            if CAMINHO_VECTORSTORE.exists():
                return InMemoryVectorStore.load(str(CAMINHO_VECTORSTORE), embedding=embeddings_model)
            return InMemoryVectorStore(embedding=embeddings_model)

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
        salvar_fingerprint()
        print(f"Fingerprint dos datasets gravado em {CAMINHO_FINGERPRINT}")
        return vectorstore
    finally:
        if lock_obtido:
            lock.release()


if __name__ == "__main__":
    vs = criar_ou_carregar_vectorstore()
    print(f"Vectorstore pronta com {len(vs.store)} documentos.")