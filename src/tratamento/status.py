import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from src.tratamento.base_vetorial import ler_fingerprint
from src.tratamento.embeddings import embeddings_model
from src.tratamento.loading import calcular_fingerprint_datasets

CAMINHO_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(CAMINHO_ENV, override=True)

CAMINHO_VECTORSTORE = Path(__file__).resolve().parents[2] / "data" / "vectorstore" / "embeddings_store.json"
PLACEHOLDER_CHAVE = "sua_chave_aqui"


def _groq_configurada() -> bool:
    chave = os.getenv("GROQ_API_KEY", "").strip()
    return bool(chave) and chave != PLACEHOLDER_CHAVE


def _modelo_fallback() -> str:
    return os.getenv("OLLAMA_FALLBACK_MODEL", "smollm2:360m")


_OLLAMA_URL = "http://localhost:11434/api/tags"
_OLLAMA_CACHE_TTL = 30.0
_ultima_verificacao_ollama = 0.0
_cache_ollama_online = False


def _ollama_online(forcar: bool = False) -> bool:
    global _ultima_verificacao_ollama, _cache_ollama_online
    agora = time.monotonic()
    if not forcar and (agora - _ultima_verificacao_ollama) < _OLLAMA_CACHE_TTL:
        return _cache_ollama_online
    try:
        with urllib.request.urlopen(_OLLAMA_URL, timeout=2) as resp:
            _cache_ollama_online = resp.status == 200
    except (urllib.error.URLError, OSError):
        _cache_ollama_online = False
    _ultima_verificacao_ollama = agora
    return _cache_ollama_online


def _contar_documentos() -> int | None:
    if not CAMINHO_VECTORSTORE.exists():
        return None
    try:
        with open(CAMINHO_VECTORSTORE, encoding="utf-8") as f:
            return len(json.load(f))
    except (OSError, json.JSONDecodeError):
        return None


def _limiar_rag() -> float:
    return float(os.getenv("RAG_LIMIAR_RELEVANCIA", "0.65"))


def _sincronia_datasets() -> tuple[str, bool | None]:
    """Compara o hash atual dos datasets com o gravado na última indexação.

    Retorna (hash_atual, sincronizado), onde sincronizado é True se bater com
    o índice, False se os dados mudaram, e None se não há referência gravada
    (índice criado antes desta feature — precisa de rebuild).
    """
    atual = calcular_fingerprint_datasets()
    indexado = ler_fingerprint()
    if indexado is None:
        return atual, None
    return atual, atual == indexado


def verificar_status(forcar_ollama: bool = False) -> dict:
    groq = _groq_configurada()
    fingerprint_atual, sincronizado = _sincronia_datasets()
    return {
        "geracao": (
            f"Groq ({os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')})"
            if groq
            else f"Fallback Ollama ({_modelo_fallback()})"
        ),
        "groq_configurada": groq,
        "ollama_online": _ollama_online(forcar=forcar_ollama),
        "documentos": _contar_documentos(),
        "rag_limiar": _limiar_rag(),
        "modelo_embedding": embeddings_model.model,
        "fallback": _modelo_fallback(),
        "agente": "RAG simples",
        "fingerprint_atual": fingerprint_atual,
        "fingerprint_indexado": ler_fingerprint(),
        "datasets_sincronizados": sincronizado,
    }
