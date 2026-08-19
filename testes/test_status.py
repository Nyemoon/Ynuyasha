from src.tratamento import base_vetorial, status


def test_ler_fingerprint_ausente_retorna_none(monkeypatch, tmp_path):
    monkeypatch.setattr(base_vetorial, "CAMINHO_FINGERPRINT", tmp_path / "inexistente.txt")
    assert base_vetorial.ler_fingerprint() is None


def test_salvar_e_ler_fingerprint(monkeypatch, tmp_path):
    caminho = tmp_path / "fingerprint.txt"
    monkeypatch.setattr(base_vetorial, "CAMINHO_FINGERPRINT", caminho)
    base_vetorial.salvar_fingerprint()
    assert caminho.exists()
    gravado = base_vetorial.ler_fingerprint()
    assert isinstance(gravado, str)
    assert len(gravado) == 64


def test_sincronia_sem_referencia(monkeypatch):
    monkeypatch.setattr(status, "calcular_fingerprint_datasets", lambda: "abc")
    monkeypatch.setattr(status, "ler_fingerprint", lambda: None)
    atual, sincronizado = status._sincronia_datasets()
    assert atual == "abc"
    assert sincronizado is None


def test_sincronia_igual(monkeypatch):
    monkeypatch.setattr(status, "calcular_fingerprint_datasets", lambda: "abc")
    monkeypatch.setattr(status, "ler_fingerprint", lambda: "abc")
    _, sincronizado = status._sincronia_datasets()
    assert sincronizado is True


def test_sincronia_desatualizada(monkeypatch):
    monkeypatch.setattr(status, "calcular_fingerprint_datasets", lambda: "novo")
    monkeypatch.setattr(status, "ler_fingerprint", lambda: "antigo")
    _, sincronizado = status._sincronia_datasets()
    assert sincronizado is False


def test_verificar_status_inclui_novos_campos(monkeypatch):
    monkeypatch.setattr(status, "_ollama_online", lambda forcar=False: False)
    monkeypatch.setattr(status, "_groq_configurada", lambda: False)
    monkeypatch.setattr(status, "calcular_fingerprint_datasets", lambda: "abc")
    monkeypatch.setattr(status, "ler_fingerprint", lambda: "abc")
    resultado = status.verificar_status()
    assert resultado["rag_limiar"] == 0.65
    assert resultado["modelo_embedding"] == "nomic-embed-text"
    assert resultado["fallback"]
    assert resultado["agente"] == "RAG simples"
    assert resultado["fingerprint_atual"] == "abc"
    assert resultado["fingerprint_indexado"] == "abc"
    assert resultado["datasets_sincronizados"] is True


def test_verificar_status_agente_sempre_rag(monkeypatch):
    monkeypatch.setattr(status, "_ollama_online", lambda forcar=False: False)
    monkeypatch.setattr(status, "_groq_configurada", lambda: True)
    monkeypatch.setattr(status, "calcular_fingerprint_datasets", lambda: "abc")
    monkeypatch.setattr(status, "ler_fingerprint", lambda: "abc")
    resultado = status.verificar_status()
    assert resultado["agente"] == "RAG simples"
