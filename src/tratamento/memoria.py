import sqlite3
import uuid
from pathlib import Path

CAMINHO_CHECKPOINTS = Path(__file__).resolve().parents[2] / "data" / "checkpoints"
CAMINHO_SQLITE = CAMINHO_CHECKPOINTS / "conversas.sqlite"

_memoria = None


def obter_memoria(checkpointer=None):
    """Retorna o checkpointer de memória do agente (criado uma única vez).

    Prioriza persistência real em SQLite (sobrevive a reinícios); cai para o
    MemorySaver em memória quando o pacote sqlite não estiver disponível ou a
    criação da conexão falhar (ex.: testes herméticos).
    """
    global _memoria
    if checkpointer is not None:
        return checkpointer
    if _memoria is None:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            CAMINHO_SQLITE.parent.mkdir(parents=True, exist_ok=True)
            _conexao = sqlite3.connect(
                str(CAMINHO_SQLITE), check_same_thread=False
            )
            _memoria = SqliteSaver(_conexao)
            _memoria.setup()
        except Exception:
            from langgraph.checkpoint.memory import MemorySaver

            _memoria = MemorySaver()
    return _memoria


def novo_thread_id() -> str:
    """Gera um identificador único de sessão/conversa."""
    return uuid.uuid4().hex
