# 🐕 Ynuyasha — Agente RAG de Astronomia

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](testes/)

> **Ynuyasha** é um agente de inteligência artificial baseado em **RAG (Retrieval-Augmented Generation)** especializado em **astronomia**. Ele consulta uma base de conhecimento construída a partir de fontes científicas reais e responde em **português do Brasil**, de forma clara, acolhedora e descritiva, com **fontes citadas**.
>
> **Status atual**: Fase M concluída — todas as features principais implementadas e testadas.

## 📋 Descrição Geral

O Ynuyasha é um assistente que roda **no terminal** e responde perguntas sobre astronomia usando apenas os dados presentes na sua base de conhecimento. Ele não inventa números ou fatos: se a informação não estiver nos dados, ele diz que não encontrou.

Para isso, o retriever filtra os trechos por um **limiar de relevância** (`RAG_LIMIAR_RELEVANCIA`, padrão **0.65**): perguntas cujo melhor trecho fique abaixo do limiar são tratadas como fora da base e o modelo responde educadamente que a informação **não consta na base de conhecimento**, sem responder com conhecimento próprio.

Além da base principal (`data/dataset/`), o agente consulta um **corpus de apoio** (`data/documentos/` — colunas do Exoplanet Archive e lista de planetas validados) via BM25 léxico. Esse corpus é **auxiliar e isolado**: não altera fingerprint, vectorstore nem exige rebuild (a base principal fica intacta); o apoio é anexado ao contexto apenas quando passa no limiar próprio `RAG_LIMIAR_APOIO_BM25` (padrão **8.0**), respeitando o teto `MAX_APOIO_CONTEXTO` (2 itens).

A recusa é imposta **em código**, não apenas no prompt: com contexto vazio (abaixo do limiar) o LLM **nem é chamado**; a resposta é a recusa padrão (`MENSAGEM_FORA_DA_BASE`).

A base de conhecimento é montada a partir de datasets reais obtidos de fontes como:

- **NASA Exoplanet Archive** — planetas e estrelas hospedeiras
- **ESA Gaia DR3** — estrelas próximas com paralaxe medida
- **SIMBAD / CDS Strasbourg** — estrelas notáveis, nebulosas e eventos extremos
- **NASA JPL Small-Body Database** — asteroides e cometas
- **Classificação própria de habitabilidade** — heurística baseada em temperatura de equilíbrio e insolação (parâmetros vindos do NASA Exoplanet Archive), inspirada em critérios usados pelo PHL (Planetary Habitability Laboratory), mas não é o algoritmo oficial do PHL
- **IAU** — constelações oficiais e definições de termos científicos

---

## 🏗️ Arquitetura Atual

```
┌──────────────┐    ┌──────────────────────┐    ┌─────────────────────────┐
│  APIs        │    │  src/consultas/      │    │  data/dataset/          │
│  Astronômicas│ ──▶│  8 scripts de coleta │ ──▶│  8 datasets (CSV)       │
└──────────────┘    └──────────────────────┘    └────────────┬────────────┘
                                                              ▼
┌──────────────┐    ┌──────────────────────┐    ┌─────────────────────────┐
│  Resposta    │ ◀──│  geração.py          │ ◀──│  retrieval.py           │
│  Markdown    │    │  LLM (Groq/Ollama)   │    │  busca semântica (k=5)  │
└──────────────┘    └──────────────────────┘    └────────────┬────────────┘
                                                              ▼
                                              ┌─────────────────────────────┐
                                              │  vectorstore (JSON, 190     │
                                              │  chunks) — embeddings       │
                                              │  Ollama nomic-embed-text    │
                                              └─────────────────────────────┘
                                              ┌─────────────────────────────┐
                                              │  apoio BM25 (isolado)       │
                                              │  RRF fusion + limiar BM25   │
                                              └─────────────────────────────┘
```

### Estrutura de diretórios

```
agente_Ynuyasha/
├── main.py                      # Interface CLI (menu, banner, status)
├── requirements.txt             # Dependências Python
├── .env                         # Variáveis de ambiente (GROQ_API_KEY etc.)
├── conftest.py                  # Configuração do pytest (raiz no sys.path)
├── interface/
│   └── app.py                   # Interface web Gradio (streaming + contexto RAG)
├── data/
│   ├── dataset/                 # CSVs gerados pelas consultas às APIs astronômicas
│   ├── documentos/              # Arquivos de apoio ao pipeline (BM25 corpus)
│   ├── vectorstore/             # Embeddings persistidos (JSON) — não versionado no git
│   └── avaliacao/               # Benchmark, resultados e logs de uso
├── src/
│   ├── consultas/               # Scripts de coleta de dados (API)
│   └── tratamento/
│       ├── loading.py           # Carrega e divide documentos em chunks (até 2000 chars, overlap 100)
│       ├── embeddings.py        # Modelo de embedding (Ollama nomic-embed-text)
│       ├── base_vetorial.py     # Cria/persiste a vectorstore (com lock anti-rebuild concorrente e checkpoints)
│       ├── retrieval.py         # Busca semântica por similaridade + merge com o apoio BM25 via RRF
│       ├── documentos_apoio.py  # Corpus de apoio (data/documentos): parsers + BM25 próprio
│       ├── geração.py           # Geração de resposta (Groq/Ollama) com pós-processamento Markdown
│       ├── agente.py            # Pipeline completo (retrieval → geração) + CLI
│       ├── avaliacao.py         # Benchmark, métricas, relatório Markdown, log/feedback
│       ├── banner.py            # Banner compartilhado (terminal + interface web)
│       ├── testar_velocidade.py # Benchmark da velocidade dos embeddings
│       └── status.py            # Diagnóstico (motor, Ollama, vectorstore, sincronia dos datasets)
└── testes/                      # Scripts de checagem de dados + testes pytest
```

### Camadas do pipeline

| Camada | Módulo | Função |
|---|---|---|
| **Coleta** | `src/consultas/` | Busca dados reais nas APIs astronômicas e gera CSVs |
| **Ingestão** | `loading.py` | Lê os CSVs e converte cada linha em texto semântico (template por dataset), com chunks de até 2000 caracteres (overlap 100) |
| **Embeddings** | `embeddings.py` | Converte textos em vetores com `nomic-embed-text` (Ollama) |
| **Indexação** | `base_vetorial.py` | Monta e persiste a vectorstore em JSON (190 chunks, com checkpoint) |
| **Retrieval** | `retrieval.py` | Busca os trechos escoreando toda a base pelos vetores gravados (sem re-embedar), funde com o BM25 por RRF e descarta os que ficam abaixo do limiar `RAG_LIMIAR_RELEVANCIA` (0.65); perguntas de enumeração (ex.: "Liste...") usam janela maior via `k_para_pergunta`; perguntas de subconjunto-por-atributo (ex.: "Quais asteroides são potencialmente perigosos?") respondem via consulta determinística ao CSV (`_FILTROS_ATRIBUTO`), sem depender do ranking vetorial |
| **Apoio** | `documentos_apoio.py` | Corpus auxiliar (`data/documentos/`) com parsers por arquivo (CSV linha a linha, sem pandas) e retriever BM25 próprio; anexa até `MAX_APOIO_CONTEXTO` itens ao contexto via `recuperar_contexto_com_apoio`, sem tocar na vectorstore principal |
| **Geração** | `geração.py` | Gera a resposta final em Markdown, citando fontes e linhas; pós-processa (`_aprimorar_markdown`) para garantir título `#`, seção `## Fontes` e organização por código |

---

## 🛠️ Tecnologias e Ferramentas

| Tecnologia | Uso |
|---|---|
| **Python 3.14** | Linguagem principal |
| **LangChain** | Orquestração do pipeline RAG (core, splitter) |
| **Ollama** | Embeddings (`nomic-embed-text`) e fallback de geração |
| **Groq** | Geração principal (`openai/gpt-oss-120b`, reasoning) |
| **Rich** | Interface visual no terminal (tabelas, painéis, Markdown) |
| **Questionary** | Menus e prompts interativos |
| **Pyfiglet** | Banner ASCII do Ynuyasha |
| **pandas / requests** | Processamento de dados e chamadas HTTP |
| **filelock** | Lock de reconstrução da vectorstore (evita rebuilds concorrentes) |

### Dependências de runtime

- **Ollama** rodando localmente com o modelo `nomic-embed-text`:
  ```bash
  ollama pull nomic-embed-text
  ```
- **Chave Groq** (opcional — sem ela, o agente usa o fallback local do Ollama):
  preencher `GROQ_API_KEY` no arquivo `.env`. O modelo padrão é
  `openai/gpt-oss-120b` (configurável via `GROQ_MODEL`); o esforço de raciocínio
  é ajustável via `GROQ_REASONING_EFFORT` (`low`/`medium`/`high`).

---

## 🛟 Fallback de geração (Ollama)

Quando `GROQ_API_KEY` não está definida no `.env` — ou quando a Groq falha durante
uma resposta — a geração usa um **modelo local do Ollama**. Os embeddings continuam
sendo feitos pelo `nomic-embed-text`; somente a **geração de texto** é que muda.

Falhas da Groq (limite de tokens/quota 429, queda de rede, instabilidade) fazem
`geração.gerar_resposta` alternar automaticamente para o fallback local. O modo do
turno (`groq`, `fallback_ollama`) é registrado nos metadados de observabilidade.

- **Padrão:** `smollm2:360m` (leve, roda em CPU).
- **Primeira vez:** baixe o modelo padrão com:
  ```bash
  ollama pull smollm2:360m
  ```
- **Trocar o modelo:** baixe outro e aponte a variável `OLLAMA_FALLBACK_MODEL` no `.env`:

  ```bash
  ollama pull qwen2.5:3b
  ```
  ```env
  # No arquivo .env (descomente e altere o valor):
  OLLAMA_FALLBACK_MODEL=qwen2.5:3b
  ```

> 💡 Modelos maiores (ex.: `qwen2.5:3b`, `llama3.2:3b`) dão respostas melhores,
> mas exigem mais memória RAM. Modelos pequenos como `smollm2:360m` são mais
> rápidos, porém com qualidade inferior à Groq.

---

## 🚀 Como Executar

### Pré-requisitos

1. **Python 3.14+** instalado.
2. **Ollama** instalado e rodando (`ollama serve`).
3. Modelo de embedding baixado: `ollama pull nomic-embed-text`.
4. **Arquivo `.env`** na raiz (veja abaixo).

### Passo a passo

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/bin/activate

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Configurar o .env (exemplo)
cp .env.example .env
# Depois edite o .env e preencha a GROQ_API_KEY (opcional)

# 4. Iniciar o Ynuyasha
python main.py
```

### Menu principal

O menu oferece as seguintes operações:

| Opção | Operação |
|---|---|
| 1 | Executar consultas (baixar/atualizar datasets) |
| 2 | (Re)construir vectorstore (Embeddings Ollama) |
| 3 | Iniciar agente interativo (conversa contínua) |
| 4 | Fazer uma pergunta única (consulta rápida) |
| 5 | Executar fluxo completo (Consultas + Vectorstore + Agente) |
| 6 | Ver diagnóstico completo do sistema |
| 7 | Iniciar interface web (Gradio) |
| 0 | Sair da aplicação |

### Comandos diretos

```bash
# Responder uma pergunta única pelo terminal
python -m src.tratamento.agente "O que é um parsec?"

# Ver o contexto recuperado para uma pergunta (sem gerar resposta)
python -m src.tratamento.retrieval "planeta em zona habitável"

# Rodar os testes automatizados (pytest, herméticos — não requerem Ollama)
python -m pytest testes/ -q

# Verificar a velocidade dos embeddings
python -m src.tratamento.testar_velocidade
```

---

## 💬 Exemplos de Respostas do Agente

> Os exemplos abaixo ilustram o estilo (Markdown, negrito nos termos-chave e
> fontes citadas). Desde a **norma descritiva**, cada dado citado ganha ainda uma
> frase a mais de explicação (o que significa e por que importa).

### Exemplo 1 — Pergunta: *"O que é um parsec?"*

> **## Definição de Parsec**
> Um **parsec (pc)** é uma unidade de medida de distância utilizada em astronomia para medir grandes distâncias no espaço.
>
> **## Explicação Simples**
> Ele equivale a cerca de **3,26 anos-luz** (ou **30 trilhões de quilômetros**). A palavra "parsec" é uma abreviação de "**paralaxe de um segundo**", referindo-se a um método trigonométrico de medir distâncias estelares.
>
> **## Definição Técnica**
> Mais tecnicamente, um parsec é definido como a distância na qual o raio da órbita da Terra (1 Unidade Astronômica) subtende um ângulo de **paralaxe de exatamente um segundo de arco**. Isso corresponde a aproximadamente **3.085677581 × 10^16 metros**.
>
> **## Unidades Relacionadas**
> As unidades de medida relacionadas ao parsec incluem:
> - **pc** (parsec)
> - **kpc** (quiloparsec)
> - **Mpc** (megaparsec)
>
> **## Fonte de Informação**
> Essas informações são baseadas em dados validados pela **IAU (International Astronomical Union)**.

---

### Exemplo 2 — Pergunta: *"Qual a temperatura de equilíbrio do TRAPPIST-1 e?"*

> **## Informações sobre o TRAPPIST-1 e**
> De acordo com os dados fornecidos, o TRAPPIST-1 e tem as seguintes características:
>
> * **Nome do Planeta:** TRAPPIST-1 e
> * **Nome da Estrela:** TRAPPIST-1
> * **Temperatura de Equilíbrio:** 249,7 K
>
> **## Resposta à Pergunta**
> A temperatura de equilíbrio do TRAPPIST-1 e é de **249,7 K**.
>
> Fonte: `habitabilidade_exoplanetas.csv`, linha 26

---

### Exemplo 3 — Pergunta: *"Quais planetas estão potencialmente na zona habitável?"*

> **## Introdução**
> A **Zona Habitável** é uma região ao redor de uma estrela onde as condições são favoráveis para a existência de vida, devido à temperatura adequada para manter água em estado líquido.
>
> **## Planetas Potencialmente Habitáveis**
> De acordo com os dados fornecidos, os seguintes planetas estão potencialmente na **Zona Habitável**:
> * **TRAPPIST-1 e**: com uma temperatura de equilíbrio de 249,7 K e um fluxo de insolação de 0,646 vezes o da Terra.
> * **TRAPPIST-1 d**: com uma temperatura de equilíbrio de 286,2 K e um fluxo de insolação de 1,115 vezes o da Terra.
>
> Ambos os planetas têm uma classificação de **Potencialmente Habitável (Conservadora)**, de acordo com os dados do **NASA Exoplanet Archive** e com a **classificação própria de habitabilidade** deste projeto (heurística por temperatura de equilíbrio, inspirada em critérios do PHL).
>
> **## Conclusão**
> Esses planetas podem ser considerados como candidatos para abrigar vida, devido às suas condições favoráveis. No entanto, é importante notar que a habitabilidade de um planeta depende de muitos fatores, incluindo a composição atmosférica, a presença de água líquida e a estabilidade climática.

---

## ❓ Exemplos de Perguntas que o Ynuyasha Responde

### Sobre o próprio agente
- Quem é você?
- O que é o Ynuyasha?
- Como você funciona?
- O que você sabe fazer?
- Qual a sua base de conhecimento?

### Exoplanetas e estrelas
- Qual o método de descoberta do Kepler-452 b?
- Qual a temperatura de equilíbrio do TRAPPIST-1 e?
- Qual o raio da estrela 55 Cnc?
- Liste planetas descobertos pelo método de trânsito.

### Zona habitável
- Quais planetas estão potencialmente na zona habitável?
- O planeta K2-18 b é habitável?
- Qual a massa do TOI-700 d?

### Glossário e conceitos
- O que é um parsec?
- Explique o método de trânsito.
- O que é a matéria escura?
- Como funciona a espectroscopia de transmissão?
- O que é um buraco negro?

### Constelações
- Qual a estrela principal de Órion?
- A constelação de Centauro é visível em qual hemisfério?
- Quantas estrelas brilhantes tem a Lira?

### Asteroides e cometas
- Quais asteroides são potencialmente perigosos?
- Qual o diâmetro do asteroide 433 Eros?
- Liste asteroides próximos da Terra.

### Estrelas e objetos do céu
- Qual o tipo de objeto M 31?
- Liste os quasares presentes na base de dados.
- Qual a paralaxe da estrela Proxima Centauri?

---

## 📊 Datasets Atuais

| Dataset | Fonte | Conteúdo |
|---|---|---|
| `planetas_e_estrelas_rag.csv` | NASA Exoplanet Archive | 31 exoplanetas e suas estrelas |
| `habitabilidade_exoplanetas.csv` | NASA Exoplanet Archive (parâmetros) + classificação própria | Habitabilidade e zona habitável (heurística por temperatura, inspirada no PHL) |
| `glossario_astronomico_conceitos.csv` | IAU / NASA | Termos científicos e definições |
| `constelacoes_iau.csv` | IAU | 7 constelações oficiais |
| `asteroides_cometas_jpl.csv` | NASA JPL | Asteroides NEOs e cometas |
| `estrelas_proximas_gaia.csv` | ESA Gaia DR3 | Estrelas próximas com paralaxe |
| `estrelas_e_objetos_simbad.csv` | SIMBAD | Estrelas notáveis e nebulosas |
| `eventos_transientes_extremos.csv` | SIMBAD | QSOs, púlsares e supernovas |

---

## 🌐 Fase 1 — Interface Web (Gradio)

Além do terminal, o Ynuyasha agora pode ser usado por um **navegador** com uma interface web feita em **Gradio**.

### Recursos

- 🖼️ **Banner do Ynuyasha** — o mesmo banner do terminal (cachorrinho + nome em ASCII colorido, via `src/tratamento/banner.py`) é exibido no topo da página.
- 💬 **Chat com streaming** — a resposta aparece token a token, renderizada em Markdown.
- 🎨 **Tela de boas-vindas com cards** — antes da primeira pergunta, um cartão de boas-vindas apresenta 4 tópicos clicáveis (parsec, método de trânsito, TRAPPIST-1 e, asteroides); clicar já envia a pergunta.
- 📋 **Copiar resposta** — cada mensagem do agente tem um botão de cópia embutido no próprio chat.
- 🔍 **Painel de contexto RAG** — os 5 pedaços recuperados (fonte, linha e relevância) ficam visíveis para auditoria; quando a pergunta não consta na base, o painel informa que nenhum trecho relevante foi encontrado.
- 💡 **Perguntas sugeridas** — exemplos clicáveis na barra lateral para começar a conversa.
- ⚙️ **Status do sistema** — painel com badges visuais: motor de geração (Groq/fallback Ollama), status do Ollama, nº de documentos na vectorstore e sincronia dos datasets.
- 🌙 **Comandos de saída** — digitar `sair`, `quit` ou `exit` encerra a conversa educadamente.

### Como executar

```bash
# A partir da raiz do projeto
python interface/app.py

# Ou com porta personalizada / link público temporário
python interface/app.py --port 7860 --share
```

A porta padrão é a **7860**; se estiver ocupada, a interface **escolhe automaticamente a próxima porta livre** (7861, 7862, ...) e imprime a URL correta no terminal — sem falhas de inicialização.

> 💡 A interface também pode ser iniciada pelo menu do CLI (**opção 7**).

---

## 🧠 Agente de IA (RAG simples)

Desde a simplificação, o Ynuyasha usa **somente o fluxo RAG clássico**: para cada
pergunta, o retriever recupera os trechos mais relevantes da base de conhecimento
(embeddings + BM25, fusão RRF) e o LLM gera a resposta ancorada nesses trechos.
Não há agente ReAct, ferramentas externas nem gate de ancoragem com parsing de
citações — o prompt instrui o modelo a responder apenas com base no contexto.

### Norma descritiva

O prompt pede que o Ynuyasha **amplie a descrição de cada informação do contexto**:
para cada número, classificação ou termo técnico citado, ele acrescenta **pelo
menos uma frase** explicando o que significa e por que importa para a pergunta
(equivalente a ≈ **25% a mais de descrição** por resposta). Isso vale apenas para
o que está no contexto — ele não inventa nem repete trechos irrelevantes.

### Respostas sobre o próprio Ynuyasha

Perguntas sobre ele mesmo (*"Quem é você?"*, *"O que é o Ynuyasha?"*,
*"Como você funciona?"*, *"Qual a sua base de conhecimento?"*) são reconhecidas
antes da recuperação: o agente responde com o próprio **conhecimento sobre si**
(identidade, dados acessados, fontes e como ele funciona) em vez de recusar como
pergunta fora da base. A detecção é feita por padrões em pt-BR
(`_e_pergunta_sobre_si` em `src/tratamento/geração.py`).

### Uso

```bash
# Terminal — uma pergunta ou conversa contínua
python -m src.tratamento.agente "Qual a definição de parsec?"
python -m src.tratamento.agente  # modo interativo
```

O painel de status mostra o modo **"RAG simples"**.

---

## 📊 Avaliação e métricas

O Ynuyasha possui uma avaliação quantitativa do pipeline, com um *benchmark*
de perguntas em pt-BR e um gerador de relatório em Markdown.

### Benchmark (`data/avaliacao/benchmark.json`)

- Perguntas cobrindo os **8 datasets** (planetas, habitabilidade,
  asteroides, constelações, glossário, SIMBAD, Gaia e eventos) com
  `linhas_esperadas` — os índices (0-based) das linhas dos CSVs que a resposta
  correta deve citar. Casos de enumeração declarada (ex.: listar todos os
  planetas de trânsito) aceitam um `k` próprio para a recuperação.
- **Casos "fora da base"** — perguntas cuja resposta correta é recusar
  educadamente (teste de honestidade).
- **Seção `apoio`** — perguntas sobre `data/documentos/` (lista de planetas
  validados e colunas do Exoplanet Archive) com `linhas_esperadas` no corpus de
  apoio; um caso pode declarar `arquivos` (lista) quando a informação é
  documentada de forma idêntica em mais de um arquivo — qualquer deles é aceito.
  Os casos "fora da base" também exigem apoio vazio (gate duplo).

### Camadas avaliadas (`src/tratamento/avaliacao.py`)

| Camada | Como | Métricas |
|---|---|---|
| **Retrieval (RAG)** | `recuperar_contexto` real, comparando metadados `(source, row)` | recall@k, precisão@k, MRR, nDCG@k, hit@1 |
| **Apoio (material de apoio)** | componente BM25 (`RecuperadorApoio.buscar`) sobre `data/documentos/`, comparando `(source, row)` | recall@k, precisão@k, MRR, nDCG@k, hit@1 |
| **Fora da base** | perguntas sem trechos relevantes não geram citações (recusa honesta) | citações / recusa |

As métricas são funções puras sobre conjuntos de chaves `(source, row)`, o que
permite testes herméticos com fakes.

### Como rodar

```bash
# A seção de retrieval é pulada sem a base vetorial/Ollama disponível;
# a seção "apoio" roda só com o corpus de data/documentos (BM25)
python -m src.tratamento.avaliacao

# Testes automatizados da avaliação (herméticos, sem Ollama)
python -m pytest testes/test_avaliacao.py -q
```

> ⚠️ A seção **Retrieval** usa embeddings do Ollama; se você não quiser
> sobrecarregar a máquina, verifique apenas os casos fora da base.

O relatório é salvo em `data/avaliacao/resultados/resultado_<timestamp>.md`
(facilita comparar a evolução entre fases).

---

## 🧠 Memória de conversa

O Ynuyasha mantém **histórico entre turnos** em memória, repassando as últimas
trocas da conversa ao LLM a cada resposta:

- **CLI** (`src/tratamento/agente.py`): o histórico da sessão é mantido pela
  própria thread; o comando `nova conversa` reinicia a memória.
- **Interface web** (`interface/app.py`): uma conversa por janela; o botão
  "Limpar conversa" reinicia a sessão.

```bash
# Testes herméticos (sem Ollama/Groq)
python -m pytest testes/ -q
```

---

## 🔒 Segurança

- Variáveis sensíveis (`GROQ_API_KEY`, etc.) ficam **apenas** no `.env`, que está no `.gitignore` e nunca vai para o repositório.
- A vectorstore (`data/vectorstore/`) também é ignorada pelo git, pois é regenerável.

---

## 📌 Status

**Fase 0 concluída** — base RAG funcional com:
- Pipeline de coleta de dados (8 fontes reais) ✔
- Vectorstore com embeddings locais (190 chunks) ✔
- Agente com geração Groq + fallback Ollama ✔
- Interface CLI completa ✔

**Fase 1 concluída** — interface web Gradio:
- Chat com streaming + Markdown ✔
- Painel de contexto RAG (fontes e relevância) ✔
- Status do sistema e perguntas sugeridas ✔
- Banner do Ynuyasha compartilhado com o terminal (ASCII colorido) ✔
- Lançamento via `python interface/app.py` e opção 7 do menu CLI ✔

**Fase H concluída** — avaliação quantitativa:
- Benchmark cobrindo os 8 datasets + casos fora da base ✔
- Métricas (recall, precisão, MRR, nDCG, hit@1) e relatório Markdown ✔
- Avaliação via RAG (retrieval) + teste de honestidade ✔

**Fase I concluída** — robustez e observabilidade (mantendo o agente simples):
- Fallback automático por erro da Groq (quota 429, rede) → Ollama local ✔
- **Recusa fora da base em código**: contexto vazio → o LLM nem é chamado;
  a resposta é a recusa padrão (sem conhecimento próprio do modelo) ✔
- Log por turno em CSV (desligado por padrão; `YNUYASHA_LOG_TURNOS=true`) ✔
- Feedback 👍/👎 na interface web → `data/avaliacao/feedback.csv` ✔
- Robustez de embeddings (timeout/keep-alive, cache de query, warm-up no boot) + lock anti-rebuild concorrente ✔

**Fase J concluída** — corpus de apoio (`data/documentos/`) e Markdown por código:
- `documentos_apoio.py`: parsers por arquivo (lista de planetas validados + 4
  CSVs de colunas do Exoplanet Archive, lidos linha a linha com o módulo `csv`) ✔
- Retriever BM25 próprio (tokenização sem stopwords, `k1=1.2`/`b=0.6`) anexado ao
  contexto via `recuperar_contexto_com_apoio`, com gate duplo
  `RAG_LIMIAR_APOIO_BM25` e teto `MAX_APOIO_CONTEXTO` — **sem invalidar a
  vectorstore principal** (fingerprint/status intactos) ✔
- **Markdown por código**: `_aprimorar_markdown` garante título `#` e a seção
  `## Fontes` com as fontes reais do contexto (`gerar_resposta` e streaming) ✔
- Benchmark com a seção `apoio` (11 casos sobre `data/documentos/` + honestidade
  do apoio) e guard test que re-deriva as linhas esperadas dos arquivos reais ✔

**Fase K concluída** — recusa honesta além do limiar (gate duplo):
- `LIMIAR_FORCA` (`RAG_LIMIAR_FORCA`, 0.68) + sobreposição lexical: ruído com
  cosseno alto mas sem termo em comum com a pergunta é descartado em código ✔
- Validação end-to-end via Groq para fatos e enumerações (paralaxe, trânsito,
  zona habitável, TRAPPIST-1 e, Kepler-452) ✔

**Fase L concluída** — retrieval determinístico (subconjuntos e fatos):
- Perguntas de atributo ("na zona habitável?", "potencialmente perigosos?",
  "quasares/púlsares/supernovas", "método de trânsito") resolvidas por filtro
  direto ao CSV, sem depender de ranking ✔
- Fatos por entidade ("O TRAPPIST-1 e está na zona habitável?") resolvidos por
  nome da entidade no arquivo — sai do problema das linhas quase idênticas ✔

**Fase M concluída** — avaliação do apoio isolada no componente:
- `avaliar_apoio` mede o próprio BM25 (`RecuperadorApoio.buscar`), não a posição
  no contexto combinado ✔
- Texto dos mapas reescrito sem preâmbulo comum (menos ruído no BM25) ✔
- Casos de colunas documentadas em mais de um arquivo aceitam `arquivos` (lista):
  pl_name (PS), disc_year, fpl_name (Composite retiring), pl_disc (Confirmed
  retiring); perguntas corrigidas para as tabelas reais ✔

**Simplificação (RAG puro):**
- Agente ReAct, ferramentas e gate de ancoragem removidos — toda resposta vem de
  `recuperar_contexto → gerar_resposta` ✔
- Prompt de sistema reescrito: natural e consistente, sem regras contraditórias ✔
- Memória de conversa em sessão (CLI/Gradio) sem checkpointer SQLite ✔

---

## 🧪 Robustez e observabilidade

O fluxo segue simples, apenas com garantias para ser honesto e resiliente:

### Recusa fora da base (não responder do conhecimento próprio)
A recusa de perguntas fora da base é **imposta em código**, não só por instrução:
**`geração.py`:** contexto vazio → o LLM nem é consultado; `gerar_resposta`
e `gerar_resposta_stream` devolvem `MENSAGEM_FORA_DA_BASE`.

### Fallback de geração
Falhas da Groq (quota 429, rede, instabilidade) fazem `geraçao.py` alternar
automaticamente para o modelo local do Ollama (`OLLAMA_FALLBACK_MODEL`).

### Log por turno (desligado por padrão)
Grava cada turno (modo, citação, latência, pergunta e resposta) em
`data/avaliacao/turnos_log.csv`. **Desligado por padrão**:

```env
# .env — ative para registrar cada turno
YNUYASHA_LOG_TURNOS=true
```

Use `false` (ou deixe sem a variável) para manter o comportamento silencioso.

### Feedback na interface web
Ao lado do chat há botões **👍/👎** que gravam a avaliação da última resposta em
`data/avaliacao/feedback.csv` — material para futuras rodadas de avaliação.

Os arquivos `turnos_log.csv` e `feedback.csv` não são versionados (dados de uso);
os relatórios de avaliação continuam em `data/avaliacao/resultados/`.