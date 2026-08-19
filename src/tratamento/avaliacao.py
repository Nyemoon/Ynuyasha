"""Avaliação quantitativa do Ynuyasha.

Roda o benchmark de `data/avaliacao/benchmark.json` nas camadas de RAG:
  1. Retrieval (RAG) — via recuperar_contexto real (exige vectorstore/Ollama),
     comparando os metadados (source, row) dos Documentos recuperados; pula com
     aviso se a base não estiver disponível;
  2. Fora da base — verifica que perguntas sem trechos relevantes não geram
     citações (honestidade).

As métricas (recall@k, precisão@k, MRR, nDCG@k, hit@1) são funções puras sobre
conjuntos/listas de chaves (source, row), o que permite testes herméticos com fakes.
O relatório em Markdown é salvo em `data/avaliacao/resultados/`.

Uso:
    python -m src.tratamento.avaliacao
"""

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
CAMINHO_BENCHMARK = RAIZ_PROJETO / "data" / "avaliacao" / "benchmark.json"
CAMINHO_RESULTADOS = RAIZ_PROJETO / "data" / "avaliacao" / "resultados"
CAMINHO_LOG_TURNOS = RAIZ_PROJETO / "data" / "avaliacao" / "turnos_log.csv"
CAMINHO_FEEDBACK = RAIZ_PROJETO / "data" / "avaliacao" / "feedback.csv"

PADRAO_CITACAO = re.compile(
    r"(?:Fonte:\s*)?(?P<arquivo>[^,;\n]+?)\s*,\s*Linha\s?(?P<linha>\d+)",
    re.IGNORECASE,
)


# ─── Métricas (funções puras) ─────────────────────────────────────────────────


def recall(esperado: set, obtido: list) -> float:
    """Fração dos itens esperados que foram recuperados.

    Sem itens esperados (fora da base), a resposta correta é não recuperar nada.
    """
    esperado = set(esperado)
    obtido = set(obtido)
    if not esperado:
        return 1.0 if not obtido else 0.0
    return len(esperado & obtido) / len(esperado)


def precisao(esperado: set, obtido: list) -> float:
    """Fração dos itens recuperados que são relevantes (citações corretas)."""
    esperado = set(esperado)
    obtido = set(obtido)
    if not obtido:
        return 1.0 if not esperado else 0.0
    return len(esperado & obtido) / len(obtido)


def mrr(esperado: set, obtido: list) -> float:
    """Reciprocal Rank: 1/posição do primeiro item relevante no ranking."""
    esperado = set(esperado)
    for posicao, item in enumerate(obtido, start=1):
        if item in esperado:
            return 1.0 / posicao
    return 0.0


def ndcg(esperado: set, obtido: list, k: int | None = None) -> float:
    """nDCG@k com ganho binário (1 se relevante, senão 0)."""
    esperado = set(esperado)
    lista = obtido[:k] if k is not None else obtido
    relevancia = [1 if item in esperado else 0 for item in lista]
    dcg = sum(rel / math.log2(pos + 2) for pos, rel in enumerate(relevancia))
    ideal = sorted(relevancia, reverse=True)
    idcg = sum(rel / math.log2(pos + 2) for pos, rel in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_at_1(esperado: set, obtido: list) -> float:
    """1 se o primeiro resultado for relevante, senão 0."""
    esperado = set(esperado)
    return 1.0 if obtido and obtido[0] in esperado else 0.0


def metricas_caso(esperado: set, obtido: list) -> dict:
    """Todas as métricas para um caso, sobre o ranking (já truncado em k)."""
    return {
        "recall": recall(esperado, obtido),
        "precisao": precisao(esperado, obtido),
        "mrr": mrr(esperado, obtido),
        "ndcg": ndcg(esperado, obtido),
        "hit@1": hit_at_1(esperado, obtido),
    }


def _media(valores: list[float]) -> float:
    return sum(valores) / len(valores) if valores else 0.0


def _resumir(metricas: list[dict]) -> dict:
    return {
        "recall": _media([m["recall"] for m in metricas]),
        "precisao": _media([m["precisao"] for m in metricas]),
        "mrr": _media([m["mrr"] for m in metricas]),
        "ndcg": _media([m["ndcg"] for m in metricas]),
        "hit@1": _media([m["hit@1"] for m in metricas]),
    }


# ─── Parsing de citações e chaves ─────────────────────────────────────────────


def extrair_citacoes(texto: str) -> list[tuple[str, int]]:
    """Extrai as citações "Fonte: <arquivo>, Linha <X>" ou "- <arquivo>, linha <N>".

    Os marcadores de lista da seção '## Fontes' (ex.: "- arquivo, linha 3") são
    removidos antes do casamento para não contaminar o nome do arquivo.
    """
    normalizado = re.sub(r"(?m)^[\s•>*\-]+(?=\S)", "", texto or "")
    return [
        (m.group("arquivo").strip(), int(m.group("linha")))
        for m in PADRAO_CITACAO.finditer(normalizado)
    ]


def citacoes_esperadas(caso: dict) -> set:
    """Chaves (arquivo, linha) esperadas; conjunto vazio para casos fora da base.

    Um caso pode declarar `arquivos` (lista) quando a informação esperada é
    documentada de forma idêntica em mais de um arquivo do corpus — aceitar
    qualquer um deles é honesto, pois a resposta citada tanto faz.
    """
    arquivos = caso.get("arquivos") or ([caso["arquivo"]] if caso.get("arquivo") else [])
    if not arquivos:
        return set()
    linhas = caso.get("linhas_esperadas", [])
    return {(arquivo, int(linha)) for arquivo in arquivos for linha in linhas}


def _chaves_documentos(resultados) -> list:
    """Extrai (source, row) dos Documentos retornados por recuperar_contexto.

    Normaliza o `source` para o nome do arquivo (basename), pois o loading grava
    o caminho completo (ex.: .../data/dataset/glossario_astronomico_conceitos.csv)
    enquanto o benchmark usa apenas o nome do arquivo.
    """
    from pathlib import Path

    chaves = []
    for doc, _score in resultados or []:
        fonte = (doc.metadata or {}).get("source")
        linha = (doc.metadata or {}).get("row")
        if fonte and linha is not None:
            chaves.append((Path(str(fonte)).name, int(linha)))
    return chaves


# ─── Avaliação: fora da base (honestidade) ────────────────────────────────────


def avaliar_fora_da_base(casos: list[dict], recuperar=None) -> dict:
    """Verifica a recusa honesta via RAG: perguntas fora da base não recuperam
    trechos relevantes (sendo assim a geração emite a recusa padrão)."""
    if recuperar is None:
        from src.tratamento.retrieval import recuperar_contexto

        recuperar = recuperar_contexto

    detalhes = []
    for caso in casos:
        pergunta = caso["pergunta"]
        try:
            resultados = recuperar(pergunta)
        except Exception as erro:
            return {"erro": str(erro)}
        citacoes = _chaves_documentos(resultados)
        recusa_honesta = not citacoes
        metricas = metricas_caso(set(), citacoes)
        detalhes.append(
            {
                "pergunta": pergunta,
                "recusa_honesta": recusa_honesta,
                "citacoes": len(citacoes),
                **metricas,
                "ok": recusa_honesta,
            }
        )
    corretos = sum(1 for d in detalhes if d["ok"])
    return {
        "resumo": {**_resumir(detalhes), "casos": len(detalhes), "corretos": corretos},
        "casos": detalhes,
    }


# ─── Avaliação: retrieval (exige vectorstore/Ollama) ──────────────────────────


def avaliar_retrieval(
    casos: list[dict], ks: tuple[int, ...] = (1, 3, 5), recuperar=None
) -> dict:
    """Avalia o RAG comparando os metadados (source, row) recuperados.

    Recupera o ranking uma única vez por pergunta e calcula as métricas @k
    sobre os prefixos do mesmo ranking — evita re-embedar a pergunta várias
    vezes. Um caso pode declarar `"k"` próprio (ex.: perguntas de enumeração
    que precisam de mais trechos); sem isso, usa o k máximo de `ks`.
    `recuperar` é injetável (padrão: recuperar_contexto), permitindo testes
    herméticos. Caso a base vetorial não esteja disponível, retorna
    {"erro": ...} e a seção é pulada no relatório.
    """
    if recuperar is None:
        from src.tratamento.retrieval import (  # noqa: PLC0415
            recuperar_contexto,
        )

        recuperar = recuperar_contexto

    from src.tratamento.retrieval import k_para_pergunta  # noqa: PLC0415

    k_max = max(ks)
    por_k = {}
    detalhes = []
    for caso in casos:
        esperado = citacoes_esperadas(caso)
        k_caso = int(
            caso.get("k") or k_para_pergunta(caso["pergunta"], k_max)
        )
        linhas_por_k = {}
        try:
            resultados = recuperar(caso["pergunta"], k=k_caso)
            chaves = _chaves_documentos(resultados)
            k_avaliar = ks if k_caso <= k_max else (k_caso,)
            for k in k_avaliar:
                sub = chaves[:k]
                linhas_por_k[k] = [linha for _a, linha in sub]
                por_k.setdefault(k, []).append(metricas_caso(esperado, sub))
        except Exception as erro:
            return {"erro": str(erro)}
        detalhes.append(
            {
                "pergunta": caso["pergunta"],
                "arquivo": caso.get("arquivo")
                or ("/".join(caso.get("arquivos") or []) or None),
                "linhas_esperadas": sorted(caso.get("linhas_esperadas", [])),
                "linhas_obtidas": linhas_por_k,
            }
        )
    resumo = {
        f"k={k}": {**_resumir(por_k[k]), "casos": len(por_k[k])}
        for k in por_k
        if por_k[k]
    }
    return {"resumo": resumo, "casos": detalhes}


# ─── Avaliação: apoio (material de apoio de data/documentos) ─────────────────


def avaliar_apoio(
    casos: list[dict], ks: tuple[int, ...] = (1, 3, 5), recuperar=None
) -> dict:
    """Avalia o componente de apoio (BM25 sobre data/documentos) por pergunta.

    Mede o próprio retriever de apoio — não o pipeline combinado — pois a
    qualificação do apoio é sobre ele encontrar o dado, não sobre a posição
    final no contexto (o pipeline combinado tem métricas próprias). `recuperar`
    é injetável para testes herméticos. Casos sem `arquivo`/`arquivos` (fora
    da base) exigem que o apoio também volte vazio.
    """
    if recuperar is None:
        from src.tratamento.documentos_apoio import (  # noqa: PLC0415
            obter_recuperador_apoio,
        )

        _apoio = obter_recuperador_apoio()

        def recuperar(pergunta: str, k: int):
            return _apoio.buscar(pergunta, top_k=k, limiar=0.0)

    return avaliar_retrieval(casos, ks=ks, recuperar=recuperar)


# ─── Relatório Markdown ───────────────────────────────────────────────────────


def _linha_md(nome: str, valor) -> str:
    return f"| {nome} | {valor} |"


def _linha_tabela(nome: str, resumo: dict) -> str:
    def valor(chave: str, casa: int | None = None) -> str:
        v = resumo.get(chave, "—")
        if isinstance(v, float) and casa is not None:
            return f"{v:.{casa}f}"
        return str(v)

    return (
        f"| {nome} | {valor('recall', 3)} | {valor('precisao', 3)} | "
        f"{valor('mrr', 3)} | {valor('ndcg', 3)} | {valor('hit@1', 3)} | "
        f"{valor('casos')} | {valor('corretos')} |"
    )


def _tabela_resumo(secoes: dict) -> str:
    linhas = [
        "| Seção | Recall | Precisão | MRR | nDCG | hit@1 | Casos | Corretos |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for nome, secao in secoes.items():
        if secao is None:
            continue
        resumo = secao.get("resumo")
        if resumo is None:
            linhas.append(_linha_tabela(nome, {}))
            continue
        if "recall" in resumo:
            linhas.append(_linha_tabela(nome, resumo))
            continue
        for k, sub in resumo.items():
            linhas.append(_linha_tabela(f"{nome} ({k})", sub))
    return "\n".join(linhas)


def _fmt_float(valor, casa=3) -> str:
    return f"{valor:.{casa}f}" if isinstance(valor, float) else str(valor)


def montar_relatorio(secoes: dict, metadados: dict) -> str:
    blocos = [
        "# Relatório de Avaliação — Ynuyasha",
        "",
        f"- **Data:** {metadados['data']}",
        f"- **Benchmark:** `{metadados['benchmark']}`",
        "",
        "## Resumo",
        "",
        _tabela_resumo(secoes),
        "",
    ]

    for nome, secao in secoes.items():
        if secao is None:
            continue
        if "erro" in secao:
            blocos += [f"## {nome}", "", f"_Não executada: {secao['erro']}_", ""]
            continue
        casos = secao.get("casos", [])
        if not casos:
            continue
        blocos.append(f"## {nome}")
        blocos.append("")
        chaves = list(casos[0].keys())
        cabecalho = "| " + " | ".join(chaves) + " |"
        separador = "|" + "---|" * len(chaves)
        blocos.append(cabecalho)
        blocos.append(separador)
        for caso in casos:
            celulas = []
            for chave in chaves:
                valor = caso[chave]
                if isinstance(valor, bool):
                    celulas.append("✔" if valor else "✘")
                elif isinstance(valor, (list, set)):
                    celulas.append(", ".join(str(v) for v in sorted(valor)))
                elif isinstance(valor, dict):
                    celulas.append(
                        "; ".join(f"{k}: {sorted(v)}" for k, v in valor.items())
                    )
                else:
                    celulas.append(_fmt_float(valor))
            blocos.append("| " + " | ".join(celulas) + " |")
        blocos.append("")
    return "\n".join(blocos).rstrip() + "\n"


def salvar_relatorio(
    secoes: dict, metadados: dict, diretorio: Path | None = None
) -> Path:
    """Salva o relatório em Markdown e devolve o caminho do arquivo."""
    destino = diretorio or CAMINHO_RESULTADOS
    destino.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = destino / f"resultado_{timestamp}.md"
    caminho.write_text(montar_relatorio(secoes, metadados), encoding="utf-8")
    return caminho


# ─── Registro de uso e feedback (Fase I) ────────────────────────────────────


def _append_csv(caminho: Path, cabecalho: list[str], linha: list) -> None:
    """Cria (com cabeçalho) e anexa uma linha em um CSV."""

    caminho.parent.mkdir(parents=True, exist_ok=True)
    novo = not caminho.exists()
    with open(caminho, "a", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        if novo:
            escritor.writerow(cabecalho)
        escritor.writerow(linha)


def registrar_turno(
    pergunta: str,
    resposta: str,
    *,
    latencia_s: float,
    citou_fonte: bool,
    modo: str,
    apoio: bool = False,
    caminho: Path | None = None,
) -> None:
    """Registra um turno de conversa em CSV (silencioso em caso de erro).

    Usado pela camada de observabilidade quando `YNUYASHA_LOG_TURNOS=true`.
    `caminho` é injetável para testes herméticos.
    """
    try:
        _append_csv(
            caminho or CAMINHO_LOG_TURNOS,
            ["timestamp", "modo", "citou_fonte", "apoio", "latencia_s", "pergunta", "resposta"],
            [
                datetime.now().isoformat(timespec="seconds"),
                modo,
                citou_fonte,
                apoio,
                f"{latencia_s:.3f}",
                (pergunta or "").replace("\n", " "),
                (resposta or "").replace("\n", " "),
            ],
        )
    except Exception:
        pass


def registrar_feedback(
    pergunta: str,
    resposta: str,
    nota: str,
    caminho: Path | None = None,
) -> None:
    """Registra o feedback do usuário (positivo/negativo) em CSV."""
    try:
        _append_csv(
            caminho or CAMINHO_FEEDBACK,
            ["timestamp", "nota", "pergunta", "resposta"],
            [
                datetime.now().isoformat(timespec="seconds"),
                nota,
                (pergunta or "").replace("\n", " "),
                (resposta or "").replace("\n", " "),
            ],
        )
    except Exception:
        pass


# ─── Execução ─────────────────────────────────────────────────────────────────


def _carregar_benchmark(caminho: Path | None = None) -> dict:
    origem = caminho or CAMINHO_BENCHMARK
    with open(origem, encoding="utf-8") as f:
        return json.load(f)


def executar_avaliacao(
    benchmark: dict | None = None, caminho: Path | None = None
) -> dict:
    """Roda as avaliações e devolve as seções para o relatório."""
    dados = benchmark or _carregar_benchmark(caminho)

    secoes = {
        "Retrieval (RAG)": avaliar_retrieval(dados.get("retrieval", [])),
        "Apoio (material de apoio)": avaliar_apoio(dados.get("apoio", [])),
        "Fora da base (honestidade)": avaliar_fora_da_base(
            dados.get("fora_da_base", [])
        ),
    }
    return secoes


def _imprimir_resumo(secoes: dict) -> None:
    for nome, secao in secoes.items():
        if secao is None:
            continue
        if "erro" in secao:
            print(f"• {nome}: não executada ({secao['erro']})")
            continue
        resumo = secao.get("resumo")
        if resumo is None:
            continue
        if "recall" in resumo:
            partes = [f"{chave}: {_fmt_float(valor)}" for chave, valor in resumo.items()]
            print(f"• {nome}: " + ", ".join(partes))
        else:
            for k, sub in resumo.items():
                partes = [f"{chave}: {_fmt_float(valor)}" for chave, valor in sub.items()]
                print(f"• {nome} ({k}): " + ", ".join(partes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação quantitativa do Ynuyasha")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="Caminho alternativo para o benchmark JSON",
    )
    args = parser.parse_args()

    secoes = executar_avaliacao(caminho=args.benchmark)

    print("=== Resumo da avaliação ===")
    _imprimir_resumo(secoes)

    metadados = {
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": args.benchmark or CAMINHO_BENCHMARK,
    }
    caminho = salvar_relatorio(secoes, metadados)
    print(f"\nRelatório salvo em: {caminho}")


if __name__ == "__main__":
    main()
