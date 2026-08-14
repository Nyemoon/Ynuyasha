import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.tratamento import agente_ia, ferramentas


class FakeLLM:
    """Emite um tool_call na 1ª chamada e responde texto na 2ª.

    A resposta padrão cita a linha do parsec (o caso usado na maioria dos
    testes) para passar no gate de ancoragem; passe `resposta` explícito para
    simular respostas não ancoradas.
    """

    def __init__(
        self,
        ferramenta="consultar_glossario",
        args=None,
        resposta="resposta final\nFonte: glossario_astronomico_conceitos.csv, Linha 1",
    ):
        self.ferramenta = ferramenta
        self.args = args or {"termo": "parsec"}
        self.resposta = resposta
        self.chamadas = 0
        self.invokes = []

    def invoke(self, mensagens):
        self.chamadas += 1
        self.invokes.append(list(mensagens))
        if self.chamadas == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": self.ferramenta,
                        "args": self.args,
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content=self.resposta)


@pytest.fixture(autouse=True)
def reseta_globais_agente():
    yield
    agente_ia._llm_agente = None
    agente_ia._grafo = None


def _monkeypatch_degrada(monkeypatch):
    chamadas = {}

    def fake_preparar(pergunta, k=5):
        chamadas["preparar"] = pergunta
        return "contexto fake"

    def fake_gerar(pergunta, contexto, historico=None):
        chamadas["gerar"] = (pergunta, contexto, historico)
        return "resposta degradada (RAG)"

    monkeypatch.setattr("src.tratamento.agente.preparar_contexto", fake_preparar)
    monkeypatch.setattr("src.tratamento.geração.gerar_resposta", fake_gerar)
    return chamadas


# ─── groq_disponivel ────────────────────────────────────────────────────────


def test_groq_disponivel_com_chave(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_qualquer_chave")
    assert agente_ia.groq_disponivel() is True


def test_groq_disponivel_ausente(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert agente_ia.groq_disponivel() is False


def test_groq_disponivel_placeholder(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sua_chave_aqui")
    assert agente_ia.groq_disponivel() is False


# ─── grafo ──────────────────────────────────────────────────────────────────


def test_criar_grafo_singleton_cacheado():
    grafo_a = agente_ia._criar_grafo()
    grafo_b = agente_ia._criar_grafo()
    assert grafo_a is grafo_b


def test_criar_grafo_injetado_nao_usa_cache():
    grafo_a = agente_ia._criar_grafo()
    grafo_b = agente_ia._criar_grafo(FakeLLM(), ferramentas.FERRAMENTAS)
    assert grafo_b is not grafo_a


# ─── executar_agente com FakeLLM ───────────────────────────────────────────


def test_executar_agente_executa_tool_e_responde():
    llm = FakeLLM()
    resultado = agente_ia.executar_agente(
        "O que é um parsec?", llm=llm, tools=ferramentas.FERRAMENTAS
    )
    assert llm.chamadas == 2
    assert resultado == "resposta final\nFonte: glossario_astronomico_conceitos.csv, Linha 1"


def test_executar_agente_inclui_toolmessage_e_system():
    llm = FakeLLM()
    agente_ia.executar_agente(
        "O que é um parsec?", llm=llm, tools=ferramentas.FERRAMENTAS
    )
    primeira = llm.invokes[0]
    assert isinstance(primeira[0], SystemMessage)
    assert agente_ia.PROMPT_AGENTE in primeira[0].content
    assert isinstance(primeira[-1], HumanMessage)
    assert primeira[-1].content == "O que é um parsec?"


def test_executar_agente_tool_executou_de_verdade():
    llm = FakeLLM()
    agente_ia.executar_agente(
        "O que é um parsec?", llm=llm, tools=ferramentas.FERRAMENTAS
    )
    segunda = llm.invokes[1]
    assert any(isinstance(m, ToolMessage) for m in segunda)
    tool_msg = next(m for m in segunda if isinstance(m, ToolMessage))
    assert tool_msg.name == "consultar_glossario"
    assert "Parsec" in tool_msg.content
    assert "Fonte: glossario_astronomico_conceitos.csv" in tool_msg.content


def test_executar_agente_tool_sem_resultado_informa():
    llm = FakeLLM(args={"termo": "zzznadaexiste"})
    agente_ia.executar_agente(
        "O que é zzznadaexiste?", llm=llm, tools=ferramentas.FERRAMENTAS
    )
    segunda = llm.invokes[1]
    tool_msg = next(m for m in segunda if isinstance(m, ToolMessage))
    assert "Nenhum" in tool_msg.content


# ─── gate de ancoragem (não responder do conhecimento próprio) ─────────────


def test_resposta_sem_citacao_vira_recusa():
    llm = FakeLLM(resposta="resposta final sem citar fonte")
    resultado = agente_ia.executar_agente(
        "O que é um parsec?", llm=llm, tools=ferramentas.FERRAMENTAS
    )
    assert resultado == agente_ia.MENSAGEM_FORA_DA_BASE


def test_resposta_com_citacao_forjada_vira_recusa():
    llm = FakeLLM(resposta="resposta com Fonte: glossario_astronomico_conceitos.csv, Linha 999")
    resultado = agente_ia.executar_agente(
        "O que é um parsec?", llm=llm, tools=ferramentas.FERRAMENTAS
    )
    assert resultado == agente_ia.MENSAGEM_FORA_DA_BASE


def test_resposta_com_citacao_plural_e_mantida():
    """Ancoragem aceita 'Linha 5' e 'Linhas 5, 7 e 10' (plural).

    Antes da relaxação, o gate só reconhecia o singular "Fonte: X, Linha N",
    então respostas corretas que agrupavam as linhas no plural eram descartadas
    e substituídas pela recusa — sintoma relatado no chat (perguntar e levar a
    resposta "fora da base de conhecimento").
    """
    llm = FakeLLM(
        ferramenta="consultar_asteroides",
        args={"termo": "potencialmente perigosos"},
        resposta=(
            "Os asteroides potencialmente perigosos são 1566 Icarus e 1620 Geographos.\n"
            "Fonte: asteroides_cometas_jpl.csv, Linhas 5 e 7"
        ),
    )
    resultado = agente_ia.executar_agente(
        "Quais asteroides são potencialmente perigosos?",
        llm=llm,
        tools=ferramentas.FERRAMENTAS,
    )
    assert resultado != agente_ia.MENSAGEM_FORA_DA_BASE


def test_resposta_com_citacao_forjada_junto_de_valida_vira_recusa():
    """Mesmo com uma citação válida, uma citação forjada mantém a recusa."""
    llm = FakeLLM(
        resposta=(
            "resposta parcial\n"
            "Fonte: glossario_astronomico_conceitos.csv, Linha 1\n"
            "Fonte: glossario_astronomico_conceitos.csv, Linha 999"
        )
    )
    resultado = agente_ia.executar_agente(
        "O que é um parsec?", llm=llm, tools=ferramentas.FERRAMENTAS
    )
    assert resultado == agente_ia.MENSAGEM_FORA_DA_BASE


def test_citacao_com_linhas_plural_ancora():
    """Citações 'Fonte: X, Linhas N, M' (plural) são extraídas e ancoram."""
    tm = ToolMessage(
        content="[1] (relevância: 0.900, fonte: glossario_astronomico_conceitos.csv, linha: 1)\n"
                "Um parsec é uma unidade de distância.\n"
                "[2] (relevância: 0.800, fonte: glossario_astronomico_conceitos.csv, linha: 2)\n"
                "Outra definição.",
        tool_call_id="call_1",
        name="buscar_na_base",
    )
    resposta = "Um parsec é uma unidade de distância.\nFonte: glossario_astronomico_conceitos.csv, Linhas 1 e 2"
    assert agente_ia._verificar_ancoragem(resposta, [tm]) is True


def test_resposta_ancorada_e_mantida():
    llm = FakeLLM(resposta="resposta ancorada\nFonte: glossario_astronomico_conceitos.csv, Linha 1")
    resultado = agente_ia.executar_agente(
        "O que é um parsec?", llm=llm, tools=ferramentas.FERRAMENTAS
    )
    assert resultado == "resposta ancorada\nFonte: glossario_astronomico_conceitos.csv, Linha 1"


def test_resposta_recusa_honesta_com_ferramenta_vazia_e_mantida():
    llm = FakeLLM(args={"termo": "zzznadaexiste"}, resposta="Não consta na base de conhecimento.")
    resultado = agente_ia.executar_agente(
        "O que é zzznadaexiste?", llm=llm, tools=ferramentas.FERRAMENTAS
    )
    assert resultado == "Não consta na base de conhecimento."


def test_resposta_sem_chamar_ferramenta_vira_recusa():
    class FakeLLMTexto:
        def invoke(self, mensagens):
            return AIMessage(content="a capital da França é Paris")

    resultado = agente_ia.executar_agente(
        "Qual a capital da França?",
        llm=FakeLLMTexto(),
        tools=ferramentas.FERRAMENTAS,
    )
    assert resultado == agente_ia.MENSAGEM_FORA_DA_BASE


def test_gate_ancoragem_aplica_com_thread_id():
    from langgraph.checkpoint.memory import MemorySaver

    cp = MemorySaver()
    llm = FakeLLM(resposta="resposta sem citação")
    resultado = agente_ia.executar_agente(
        "O que é um parsec?", llm=llm, tools=ferramentas.FERRAMENTAS,
        checkpointer=cp, thread_id="sessao_ancoragem",
    )
    assert resultado == agente_ia.MENSAGEM_FORA_DA_BASE


# ─── buscar_na_base: formato RAG (fonte:/linha:) no retorno de ferramentas ───


def test_buscar_na_base_formato_rag_ancora_citacao_canonica():
    tm = ToolMessage(
        content="[1] (relevância: 0.870, fonte: glossario_astronomico_conceitos.csv, linha: 1)\n"
                "Um parsec é uma unidade de distância.",
        tool_call_id="call_1",
        name="buscar_na_base",
    )
    resposta = "Um parsec é uma unidade de distância.\nFonte: glossario_astronomico_conceitos.csv, Linha 1"
    assert agente_ia._verificar_ancoragem(resposta, [tm]) is True


def test_buscar_na_base_formato_rag_exige_citacao_canonica_na_resposta():
    tm = ToolMessage(
        content="[1] (relevância: 0.870, fonte: glossario_astronomico_conceitos.csv, linha: 1)\n"
                "Um parsec é uma unidade de distância.",
        tool_call_id="call_1",
        name="buscar_na_base",
    )
    resposta = "Um parsec é uma unidade de distância.\nfonte: glossario_astronomico_conceitos.csv, linha: 1"
    assert agente_ia._verificar_ancoragem(resposta, [tm]) is False


def test_executar_agente_ancora_citacao_de_toolmessage_rag():
    """Uma ToolMessage no formato RAG (buscar_na_base) ancora a resposta.

    Não executa a ferramenta real (que depende de Ollama) nem o grafo; valida o
    gate de ancoragem de ponta a ponta no agente injetando a ToolMessage pronta
    no loop, para não depender do ToolNode real.
    """
    tool_msg = ToolMessage(
        content="[1] (relevância: 0.870, fonte: glossario_astronomico_conceitos.csv, linha: 1)\n"
                "Um parsec é uma unidade de distância.",
        tool_call_id="call_1",
        name="buscar_na_base",
    )
    resposta = "resposta via busca livre\nFonte: glossario_astronomico_conceitos.csv, Linha 1"
    assert agente_ia._verificar_ancoragem(resposta, [tool_msg]) is True

    resposta_sem_citacao = "resposta via busca livre sem citar fonte"
    assert agente_ia._verificar_ancoragem(resposta_sem_citacao, [tool_msg]) is False


# ─── ancoragem com memória: só vale o que as ferramentas devolveram no turno ───


def test_thread_nao_ancora_com_citacao_de_turno_anterior():
    from langgraph.checkpoint.memory import MemorySaver

    class FakeMultiTurn:
        def __init__(self):
            self.n = 0

        def invoke(self, mensagens):
            self.n += 1
            if self.n == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "consultar_glossario",
                        "args": {"termo": "parsec"},
                        "id": "call_1",
                        "type": "tool_call",
                    }],
                )
            if self.n == 2:
                return AIMessage(
                    content="resposta turno 1\nFonte: glossario_astronomico_conceitos.csv, Linha 1"
                )
            return AIMessage(
                content="a capital da França é Paris\nFonte: glossario_astronomico_conceitos.csv, Linha 1"
            )

    cp = MemorySaver()
    llm = FakeMultiTurn()

    r1 = agente_ia.executar_agente(
        "O que é parsec?", llm=llm, tools=ferramentas.FERRAMENTAS,
        checkpointer=cp, thread_id="sessao_multiturn",
    )
    assert "resposta turno 1" in r1

    r2 = agente_ia.executar_agente(
        "Qual a capital da França?", llm=llm, tools=ferramentas.FERRAMENTAS,
        checkpointer=cp, thread_id="sessao_multiturn",
    )
    assert r2 == agente_ia.MENSAGEM_FORA_DA_BASE


def test_thread_ancora_com_ferramenta_do_turno_atual():
    from langgraph.checkpoint.memory import MemorySaver

    class FakeReActTodaChamada:
        def __init__(self):
            self.n = 0

        def invoke(self, mensagens):
            self.n += 1
            if self.n in (1, 3):
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "consultar_glossario",
                        "args": {"termo": "parsec"},
                        "id": "call_1",
                        "type": "tool_call",
                    }],
                )
            return AIMessage(
                content=f"resposta turno {(self.n + 1) // 2}\n"
                        f"Fonte: glossario_astronomico_conceitos.csv, Linha 1"
            )

    cp = MemorySaver()
    llm = FakeReActTodaChamada()

    agente_ia.executar_agente(
        "O que é parsec?", llm=llm, tools=ferramentas.FERRAMENTAS,
        checkpointer=cp, thread_id="sessao_ok",
    )
    r2 = agente_ia.executar_agente(
        "E isso vale pra quê?", llm=llm, tools=ferramentas.FERRAMENTAS,
        checkpointer=cp, thread_id="sessao_ok",
    )
    assert "resposta turno 2" in r2


# ─── degrade sem Groq ──────────────────────────────────────────────────────


def test_executar_agente_degrada_sem_groq(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    chamadas = _monkeypatch_degrada(monkeypatch)

    resultado = agente_ia.executar_agente("O que é um parsec?")
    assert resultado == "resposta degradada (RAG)"
    assert chamadas["preparar"] == "O que é um parsec?"
    assert chamadas["gerar"][1] == "contexto fake"


def test_executar_agente_degrada_repassa_historico(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    chamadas = _monkeypatch_degrada(monkeypatch)
    historico = [{"role": "user", "content": "oi"}, {"role": "assistant", "content": "olá"}]

    agente_ia.executar_agente("Pergunta?", historico=historico)
    assert chamadas["gerar"][2] == historico


def test_executar_agente_fallback_quando_groq_falha(monkeypatch):
    """Falha da Groq (quota 429) degrada para RAG/Ollama em vez de propagar.

    Reproduz o sintoma relatado: com a quota diária da Groq esgotada, cada
    pergunta do chat quebrava com RateLimitError e o usuário ficava sem resposta.
    """
    chamadas = {}

    def fake_degrada(pergunta, historico=None):
        chamadas["pergunta"] = pergunta
        chamadas["historico"] = historico
        return "resposta do fallback (RAG/Ollama)"

    monkeypatch.setattr(agente_ia, "_degradar_rag", fake_degrada)

    class RateLimitErrorStub(Exception):
        pass

    class LLMRateLimit:
        def invoke(self, mensagens):
            raise RateLimitErrorStub("quota esgotada")

    resultado = agente_ia.executar_agente(
        "O que é um parsec?",
        llm=LLMRateLimit(),
        tools=ferramentas.FERRAMENTAS,
    )
    assert resultado == "resposta do fallback (RAG/Ollama)"
    assert chamadas["pergunta"] == "O que é um parsec?"


def test_executar_agente_fallback_registra_modo_fallback(monkeypatch):
    """O registrador recebe modo='fallback_ollama' quando a Groq falha."""
    recebido = {}
    monkeypatch.setattr(
        agente_ia, "_degradar_rag", lambda pergunta, historico=None: "fallback"
    )

    class LLMRateLimit:
        def invoke(self, mensagens):
            raise RuntimeError("quota esgotada")

    agente_ia.executar_agente(
        "Pergunta?",
        llm=LLMRateLimit(),
        tools=ferramentas.FERRAMENTAS,
        registrador=lambda md: recebido.update(md),
    )
    assert recebido["modo"] == "fallback_ollama"
    assert recebido["tool_calls"] == 0


def test_executar_agente_com_llm_injetado_nao_degrada(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    _monkeypatch_degrada(monkeypatch)
    llm = FakeLLM()

    resultado = agente_ia.executar_agente(
        "Pergunta?", llm=llm, tools=ferramentas.FERRAMENTAS
    )
    assert llm.chamadas == 2
    assert resultado == "resposta final\nFonte: glossario_astronomico_conceitos.csv, Linha 1"


def test_executar_agente_com_groq_sem_llm_usa_obter_llm(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_qualquer_chave")
    agente_ia._llm_agente = None
    llm = agente_ia._obter_llm()
    assert hasattr(llm, "invoke")
    assert hasattr(llm, "bound")


# ─── poda de memória (Fase I) ────────────────────────────────────────────────


def test_ultimas_mensagens_mantem_system():
    msgs = [SystemMessage(content="sys"), HumanMessage(content="a"), AIMessage(content="b"),
            HumanMessage(content="c"), AIMessage(content="d")]
    podada = agente_ia._ultimas_mensagens(msgs, limite=2)
    assert isinstance(podada[0], SystemMessage)
    assert isinstance(podada[1], HumanMessage)
    assert [m.content for m in podada] == ["sys", "c", "d"]


def test_ultimas_mensagens_sem_corte():
    msgs = [SystemMessage(content="sys"), HumanMessage(content="a")]
    assert agente_ia._ultimas_mensagens(msgs, limite=5) is msgs


def test_ultimas_mensagens_corte_exato():
    msgs = [SystemMessage(content="sys"), HumanMessage(content="a"), AIMessage(content="b")]
    podada = agente_ia._ultimas_mensagens(msgs, limite=2)
    assert [m.content for m in podada] == ["sys", "a", "b"]


def test_thread_longo_limita_mensagens_enviadas():
    class FakeLLMResposta:
        def __init__(self):
            self.chamadas = 0
            self.invokes = []

        def invoke(self, mensagens):
            self.chamadas += 1
            self.invokes.append(list(mensagens))
            return AIMessage(content=f"resposta {self.chamadas}")

    from langgraph.checkpoint.memory import MemorySaver

    llm = FakeLLMResposta()
    cp = MemorySaver()
    thread = "sessao_longa"

    for i in range(5):
        agente_ia.executar_agente(
            f"pergunta {i}", llm=llm, tools=ferramentas.FERRAMENTAS,
            checkpointer=cp, thread_id=thread,
        )

    ultima_envio = llm.invokes[-1]
    assert sum(isinstance(m, SystemMessage) for m in ultima_envio) == 1
    assert len(ultima_envio) <= agente_ia.LIMITE_MENSAGENS_AGENTE + 1


# ─── fallback automático (Fase I) ────────────────────────────────────────────


def test_prompt_agente_inclui_fallback_buscar_na_base():
    assert "buscar_na_base" in agente_ia.PROMPT_AGENTE
    assert "Nenhum registro" in agente_ia.PROMPT_AGENTE


# ─── registrador (Fase I) ────────────────────────────────────────────────────


def test_executar_agente_registrador_recebe_metadados():
    llm = FakeLLM(resposta="resposta com Fonte: glossario_astronomico_conceitos.csv, Linha 1")
    recebido = {}
    chamadas_registrador = []

    def registrador(md):
        chamadas_registrador.append(md)
        recebido.update(md)

    agente_ia.executar_agente(
        "O que é um parsec?", llm=llm, tools=ferramentas.FERRAMENTAS,
        registrador=registrador,
    )
    assert chamadas_registrador
    assert recebido["modo"] == "agente"
    assert recebido["tool_calls"] == 1
    assert "latencia_s" in recebido
    assert recebido["citou_fonte"] is True


def test_executar_agente_registrador_detecta_sem_citacao():
    llm = FakeLLM(resposta="resposta sem citar fonte")
    recebido = {}

    agente_ia.executar_agente(
        "O que é um parsec?", llm=llm, tools=ferramentas.FERRAMENTAS,
        registrador=lambda md: recebido.update(md),
    )
    assert recebido["citou_fonte"] is False


def test_executar_agente_degrada_registrador_recebe_rag(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    chamadas = _monkeypatch_degrada(monkeypatch)
    recebido = {}

    agente_ia.executar_agente(
        "O que é um parsec?", registrador=lambda md: recebido.update(md)
    )
    assert recebido["modo"] == "rag"
    assert recebido["tool_calls"] == 0
    assert chamadas["preparar"] == "O que é um parsec?"


def test_executar_agente_sem_registrador_nao_falha():
    llm = FakeLLM()
    resultado = agente_ia.executar_agente(
        "O que é um parsec?", llm=llm, tools=ferramentas.FERRAMENTAS,
        registrador=None,
    )
    assert resultado == "resposta final\nFonte: glossario_astronomico_conceitos.csv, Linha 1"
