from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.tratamento import geração


class FakeLLM:
    def __init__(self, texto="resposta fake", levantar=False):
        self._texto = texto
        self._levantar = levantar

    def invoke(self, prompt):
        if self._levantar:
            raise RuntimeError("erro fake")
        return SimpleNamespace(content=self._texto)

    def stream(self, prompt):
        if self._levantar:
            raise RuntimeError("erro fake")
        yield SimpleNamespace(content=self._texto)


@pytest.fixture(autouse=True)
def reseta_estado_global():
    yield
    geração._llm = None
    geração._usando_groq = False


def test_historico_para_mensagens_converte_roles():
    mensagens = geração._historico_para_mensagens(
        [
            {"role": "user", "content": "oi"},
            {"role": "assistant", "content": "olá"},
        ]
    )
    assert isinstance(mensagens[0], HumanMessage)
    assert mensagens[0].content == "oi"
    assert isinstance(mensagens[1], AIMessage)
    assert mensagens[1].content == "olá"


def test_historico_para_mensagens_ignora_papeis_desconhecidos():
    mensagens = geração._historico_para_mensagens(
        [{"role": "system", "content": "ignorado"}]
    )
    assert mensagens == []


def test_historico_para_mensagens_none():
    assert geração._historico_para_mensagens(None) == []


def test_montar_prompt_estrutura_completa():
    mensagens = geração.montar_prompt(
        "O que é um parsec?",
        "contexto relevante",
        historico=[{"role": "user", "content": "oi"}],
    )
    assert isinstance(mensagens[0], SystemMessage)
    assert mensagens[0].content == geração.SISTEMA_ASTRONOMIA
    assert len(mensagens) == 3
    assert isinstance(mensagens[1], HumanMessage)
    assert isinstance(mensagens[2], HumanMessage)
    ultimo = mensagens[-1].content
    assert "CONTEXTO:" in ultimo
    assert "PERGUNTA DO USUÁRIO:" in ultimo
    assert "contexto relevante" in ultimo
    assert "O que é um parsec?" in ultimo


def test_montar_prompt_sem_historico():
    mensagens = geração.montar_prompt("pergunta", "contexto")
    assert len(mensagens) == 2
    assert isinstance(mensagens[1], HumanMessage)


def test_montar_prompt_contexto_vazio_usa_marcador():
    mensagens = geração.montar_prompt("pergunta", "   ")
    ultimo = mensagens[-1].content
    assert "Nenhum trecho relevante encontrado" in ultimo


def test_gerar_resposta_com_fake_llm(monkeypatch):
    monkeypatch.setattr(geração, "obter_llm", lambda: FakeLLM(texto="resposta fake"))
    resposta = geração.gerar_resposta("pergunta", "contexto")
    assert resposta == "# pergunta\n\nresposta fake"


def test_gerar_resposta_contexto_vazio_nao_chama_llm(monkeypatch):
    def obter_llm_falha():
        raise AssertionError("LLM não deveria ser consultado sem contexto")

    monkeypatch.setattr(geração, "obter_llm", obter_llm_falha)
    resposta = geração.gerar_resposta("pergunta", "   ")
    assert resposta == geração.MENSAGEM_FORA_DA_BASE


def test_gerar_resposta_contexto_vazio_recusa_sem_invocar_llm(monkeypatch):
    chamadas = []

    def obter_llm_contador():
        chamadas.append(1)
        return FakeLLM(texto="resposta fake")

    monkeypatch.setattr(geração, "obter_llm", obter_llm_contador)
    resposta = geração.gerar_resposta("pergunta", "")
    assert resposta == geração.MENSAGEM_FORA_DA_BASE
    assert chamadas == []


def test_gerar_resposta_usa_fallback_apos_erro_groq(monkeypatch):
    falha = FakeLLM(levantar=True)
    monkeypatch.setattr(geração, "obter_llm", lambda: falha)
    monkeypatch.setattr(
        geração, "_criar_llm_fallback", lambda: FakeLLM(texto="resposta local")
    )
    geração._usando_groq = True

    resposta = geração.gerar_resposta("pergunta", "contexto")
    assert resposta == "# pergunta\n\nresposta local"
    assert geração._usando_groq is False


def test_gerar_resposta_relanca_erro_sem_fallback(monkeypatch):
    falha = FakeLLM(levantar=True)
    monkeypatch.setattr(geração, "obter_llm", lambda: falha)
    geração._usando_groq = False

    with pytest.raises(RuntimeError):
        geração.gerar_resposta("pergunta", "contexto")


def test_gerar_resposta_stream_com_fake_llm(monkeypatch):
    monkeypatch.setattr(geração, "obter_llm", lambda: FakeLLM(texto="resposta fake"))
    texto = "".join(geração.gerar_resposta_stream("pergunta", "contexto"))
    assert texto == "# pergunta\n\nresposta fake"


def test_gerar_resposta_stream_contexto_vazio_nao_chama_llm(monkeypatch):
    def obter_llm_falha():
        raise AssertionError("LLM não deveria ser consultado sem contexto")

    monkeypatch.setattr(geração, "obter_llm", obter_llm_falha)
    texto = "".join(geração.gerar_resposta_stream("pergunta", ""))
    assert texto == geração.MENSAGEM_FORA_DA_BASE


def test_gerar_resposta_stream_usa_fallback(monkeypatch):
    falha = FakeLLM(levantar=True)
    monkeypatch.setattr(geração, "obter_llm", lambda: falha)
    monkeypatch.setattr(
        geração, "_criar_llm_fallback", lambda: FakeLLM(texto="resposta local")
    )
    geração._usando_groq = True

    texto = "".join(geração.gerar_resposta_stream("pergunta", "contexto"))
    assert texto == "# pergunta\n\nresposta local"
    assert geração._usando_groq is False


# ─── Markdown por código (_aprimorar_markdown) ────────────────────────────────


def test_coletar_fontes_extrai_blocos_do_contexto():
    contexto = (
        "[1] (relevância: 0.90, fonte: glossario.csv, linha: 1)\nparsec\n\n"
        "[2] (relevância: 0.80, fonte: planetas_e_estrelas_rag.csv, linha: 23)\nTRAPPIST"
    )
    assert geração._coletar_fontes(contexto) == [
        ("glossario.csv", "1"),
        ("planetas_e_estrelas_rag.csv", "23"),
    ]


def test_coletar_fontes_remove_duplicatas():
    contexto = "[1] (fonte: a.csv, linha: 1)\nx\n\n[2] (fonte: a.csv, linha: 1)\ny"
    assert geração._coletar_fontes(contexto) == [("a.csv", "1")]


def test_coletar_fontes_sem_contexto():
    assert geração._coletar_fontes("") == []
    assert geração._coletar_fontes(None) == []


def test_aprimorar_markdown_adiciona_titulo_e_fontes():
    contexto = "[1] (fonte: a.csv, linha: 3)\nburaco negro"
    texto = geração._aprimorar_markdown(
        "resposta corrida", contexto, "O que é um parsec?"
    )
    assert texto.startswith("# O que é um parsec")
    assert "## Fontes" in texto
    assert "- a.csv, linha 3" in texto


def test_aprimorar_markdown_nao_duplica_titulo_existente():
    texto = geração._aprimorar_markdown(
        "## Já tem título\ncorpo", "", "pergunta"
    )
    assert texto.startswith("## Já tem título")


def test_aprimorar_markdown_vazio_retorna_vazio():
    assert geração._aprimorar_markdown("   ", "", "") == ""


def test_aprimorar_markdown_sem_contexto_nao_anexa_fontes():
    texto = geração._aprimorar_markdown("resposta", "", "pergunta")
    assert "## Fontes" not in texto


# ─── Fontes seletivas (a partir das citações do modelo) ───────────────────────

CONTEXTO_FONTES = (
    "[1] (relevância: 0.73, fonte: estrelas_e_objetos_simbad.csv, linha: 7)\n"
    "O objeto astronômico NAME Proxima Centauri...\n\n"
    "[2] (relevância: 0.76, fonte: constelacoes_iau.csv, linha: 1)\n"
    "A constelação de Centauro..."
)


def test_indices_citados_extrai_marcadores():
    texto = "Resposta【1】com citação【1†linha 7】e【2】."
    assert geração._indices_citados(texto) == {1, 2}
    assert geração._indices_citados("sem marcadores") == set()


def test_fontes_do_contexto_mapeia_indice_para_fonte():
    mapa = geração._fontes_do_contexto(CONTEXTO_FONTES)
    assert mapa[1] == ("estrelas_e_objetos_simbad.csv", "7")
    assert mapa[2] == ("constelacoes_iau.csv", "1")


def test_montar_fontes_markdown_filtra_pelos_indices_citados():
    fontes = geração._montar_fontes_markdown(CONTEXTO_FONTES, indices={1})
    assert "- estrelas_e_objetos_simbad.csv, linha 7" in fontes
    assert "constelacoes_iau.csv" not in fontes


def test_montar_fontes_markdown_indices_invalidos_cai_para_todas():
    fontes = geração._montar_fontes_markdown(CONTEXTO_FONTES, indices={9})
    assert "estrelas_e_objetos_simbad.csv, linha 7" in fontes
    assert "constelacoes_iau.csv, linha 1" in fontes


def test_montar_fontes_markdown_usa_citacao_inline_sem_marcadores():
    corpo = "A paralaxe é 768.067 mas. **Fonte:** estrelas_e_objetos_simbad.csv, linha 7"
    fontes = geração._montar_fontes_markdown(CONTEXTO_FONTES, indices=set(), corpo=corpo)
    assert "- estrelas_e_objetos_simbad.csv, linha 7" in fontes
    assert "constelacoes_iau.csv" not in fontes


def test_aprimorar_markdown_remove_secao_de_fontes_do_modelo():
    contexto = "[1] (relevância: 0.73, fonte: a.csv, linha: 7)\nx"
    texto = geração._aprimorar_markdown(
        "Corpo. \n### Fontes\n- a.csv, linha 7\n", contexto, "pergunta"
    )
    assert "### Fontes" not in texto
    assert "## Fontes" in texto
    assert texto.count("## Fontes") == 1


def test_aprimorar_markdown_nao_cria_fontes_duplicadas():
    contexto = "[1] (relevância: 0.73, fonte: a.csv, linha: 7)\nx"
    texto = geração._aprimorar_markdown(
        "**Fonte:** a.csv, linha 7【1】", contexto, "pergunta"
    )
    assert texto.count("## Fontes") == 1


def test_sistema_astronomia_impoe_novas_normas():
    assert "estritamente à pergunta feita" in geração.SISTEMA_ASTRONOMIA
    assert "NUNCA repita" in geração.SISTEMA_ASTRONOMIA
    assert "listagem ou enumeração" in geração.SISTEMA_ASTRONOMIA


def test_sistema_astronomia_exige_descricao_ampliada():
    assert "pelo menos uma frase" in geração.SISTEMA_ASTRONOMIA
    assert "o que significa e por que importa para a pergunta" in geração.SISTEMA_ASTRONOMIA


# ─── Perguntas sobre o próprio Ynuyasha ───────────────────────────────────────


def test_e_pergunta_sobre_si_reconhece_padroes():
    for pergunta in (
        "Quem é você?",
        "O que é o Ynuyasha?",
        "Como você funciona?",
        "Fale sobre você",
        "Me conte sobre o Ynuyasha",
        "Qual a sua função?",
        "O que você sabe fazer?",
        "Qual a sua base de conhecimento?",
        "Você é um agente de astronomia?",
    ):
        assert geração._e_pergunta_sobre_si(pergunta), pergunta


def test_e_pergunta_sobre_si_nao_reconhece_outras():
    for pergunta in (
        "O que é um parsec?",
        "Qual a capital da França?",
        "Como você está?",
        "O que você sabe sobre a Lua?",
        "Qual o método de descoberta do Kepler-452 b?",
    ):
        assert not geração._e_pergunta_sobre_si(pergunta), pergunta


def test_gerar_resposta_pergunta_sobre_si_usa_contexto_proprio(monkeypatch):
    capturados = {}

    def fake_invoke(prompt):
        capturados["prompt"] = prompt
        return SimpleNamespace(content="Eu sou o Ynuyasha.")

    monkeypatch.setattr(
        geração, "obter_llm", lambda: SimpleNamespace(invoke=fake_invoke)
    )
    resposta = geração.gerar_resposta("Quem é você?", "")
    assert "Eu sou o Ynuyasha" in resposta
    corpo = capturados["prompt"][-1].content
    assert geração.SOBRE_YNUVASHA in corpo


def test_gerar_resposta_stream_pergunta_sobre_si_nao_recusa(monkeypatch):
    class FakeLLM:
        def stream(self, prompt):
            yield SimpleNamespace(content="Sobre mim: sou o Ynuyasha.")

    monkeypatch.setattr(geração, "obter_llm", lambda: FakeLLM())
    texto = "".join(geração.gerar_resposta_stream("Quem é você?", ""))
    assert geração.MENSAGEM_FORA_DA_BASE not in texto
    assert "Ynuyasha" in texto


def test_gerar_resposta_pergunta_comum_contexto_vazio_continua_recusando(monkeypatch):
    def obter_llm_falha():
        raise AssertionError("LLM não deveria ser consultado sem contexto")

    monkeypatch.setattr(geração, "obter_llm", obter_llm_falha)
    resposta = geração.gerar_resposta("Qual a capital da França?", "")
    assert resposta == geração.MENSAGEM_FORA_DA_BASE
