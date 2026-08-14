# 🐕 Ynuyasha — Agente RAG de Astronomia

> **Ynuyasha** é um agente de inteligência artificial baseado em **RAG (Retrieval-Augmented Generation)** especializado em **astronomia**. Ele consulta uma base de conhecimento construída a partir de fontes científicas reais e responde em **português do Brasil**, de forma clara, objetiva e com **fontes citadas**.

---

## 📋 Descrição Geral

O Ynuyasha é um assistente que roda **no terminal** e responde perguntas sobre astronomia usando apenas os dados presentes na sua base de conhecimento. Ele não inventa números ou fatos: se a informação não estiver nos dados, ele diz que não encontrou.

Para isso, o retriever filtra os trechos por um **limiar de relevância** (`RAG_LIMIAR_RELEVANCIA`, padrão **0.65**): perguntas cujo melhor trecho fique abaixo do limiar são tratadas como fora da base e o modelo responde educadamente que a informação **não consta na base de conhecimento**, sem responder com conhecimento próprio.

A recusa é imposta **em código**, não apenas no prompt (gate de ancoragem):

- **Caminho RAG:** contexto vazio (abaixo do limiar) → o LLM **nem é chamado**; a resposta é a recusa padrão (`MENSAGEM_FORA_DA_BASE`).
- **Caminho agente ReAct (Groq):** após o loop de ferramentas, a resposta só é aceita se estiver **ancorada** nos dados retornados pelas ferramentas — precisa conter **ao menos uma** citação `Fonte: <arquivo>, Linha X` que exista no retorno de alguma ferramenta (o retorno aceita também o formato RAG `fonte: <arquivo>, linha: <X>` emitido por `buscar_na_base`; citações no plural como `Fonte: X, Linhas 5, 7 e 10` são suportadas). Se **qualquer** citação for forjada (não estiver no retorno das ferramentas), a resposta é substituída pela recusa. Respostas sem nenhuma citação (com dados disponíveis) ou sem nenhum dado de ferramenta também são **substituídas pela recusa**. Em conversas com `thread_id`, apenas as ferramentas chamadas **no turno atual** ancoram a resposta. Isso vale também em conversas com `thread_id` (memória persistente).

A base de conhecimento é montada a partir de datasets reais obtidos de fontes como:

- **NASA Exoplanet Archive** — planetas e estrelas hospedeiras
- **ESA Gaia DR3** — estrelas próximas com paralaxe medida
- **SIMBAD / CDS Strasbourg** — estrelas notáveis, nebulosas e eventos extremos
- **NASA JPL Small-Body Database** — asteroides e cometas
- **PHL (Planetary Habitability Laboratory)** — estimativas de zona habitável
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
│   ├── dataset/                 # CSVs gerados pelas consultas
│   ├── documentos/              # Arquivos de apoio ao pipeline
│   └── vectorstore/             # Embeddings persistidos (JSON)
├── src/
│   ├── consultas/               # Scripts de coleta de dados (API)
│   └── tratamento/
│       ├── loading.py           # Carrega e divide documentos
│       ├── embeddings.py        # Modelo de embedding (Ollama)
│       ├── base_vetorial.py     # Cria/persiste a vectorstore (com lock anti-rebuild concorrente e checkpoints)
│       ├── retrieval.py         # Busca semântica por similaridade
│       ├── geração.py           # Geração de resposta (Groq/Ollama)
│       ├── agente.py            # Pipeline completo (retrieval → geração)
│       ├── agente_ia.py         # Agente ReAct com tool-calling (grafo LangGraph + memória)
│       ├── ferramentas.py       # 9 ferramentas que leem os CSVs diretamente
│       ├── memoria.py           # Checkpointer SQLite/MemorySaver + novo_thread_id
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
| **Retrieval** | `retrieval.py` | Busca os trechos fundindo busca vetorial + BM25 (RRF); re-escoreia com embeddings apenas o topo dos candidatos só-BM25 (`MAX_REESCORE`) e descarta os que ficam abaixo do limiar `RAG_LIMIAR_RELEVANCIA` (0.65) |
| **Geração** | `geração.py` | Gera a resposta final em Markdown, citando fontes e linhas |

---

## 🛠️ Tecnologias e Ferramentas

| Tecnologia | Uso |
|---|---|
| **Python 3.14** | Linguagem principal |
| **LangChain** | Orquestração do pipeline RAG (core, splitter) |
| **Ollama** | Embeddings (`nomic-embed-text`) e fallback de geração |
| **Groq** | Geração principal (`llama-3.3-70b-versatile`) |
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
  preencher `GROQ_API_KEY` no arquivo `.env`.

---

## 🛟 Fallback de geração (Ollama)

Quando `GROQ_API_KEY` não está definida no `.env` — ou quando a Groq falha durante
uma resposta — a geração usa um **modelo local do Ollama**. Os embeddings continuam
sendo feitos pelo `nomic-embed-text`; somente a **geração de texto** é que muda.

No caminho do agente ReAct (`agente_ia.executar_agente`), falhas da Groq são
tratadas com **fallback automático**: qualquer exceção (limite de tokens/quota
429, `tool_use_failed` 400, queda de rede) faz o turno degradar para o fluxo RAG
local (`_degradar_rag` → `preparar_contexto` + `gerar_resposta`), que por sua vez
tem fallback Groq→Ollama. O modo do turno (`groq`, `fallback_ollama`) é
registrado nos metadados de observabilidade.

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
> Ambos os planetas têm uma classificação de **Potencialmente Habitável (Conservadora)**, de acordo com os dados do **NASA Exoplanet Archive** e do **Algoritmo PHL**.
>
> **## Conclusão**
> Esses planetas podem ser considerados como candidatos para abrigar vida, devido às suas condições favoráveis. No entanto, é importante notar que a habitabilidade de um planeta depende de muitos fatores, incluindo a composição atmosférica, a presença de água líquida e a estabilidade climática.

---

## ❓ Exemplos de Perguntas que o Ynuyasha Responde

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
| `habitabilidade_exoplanetas.csv` | NASA / PHL | Habitabilidade e zona habitável |
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
- 🔍 **Painel de contexto RAG** — os 5 pedaços recuperados (fonte, linha e relevância) ficam visíveis para auditoria; quando a pergunta não consta na base, o painel informa que nenhum trecho relevante foi encontrado.
- 💡 **Perguntas sugeridas** — exemplos clicáveis para começar a conversa.
- ⚙️ **Status do sistema** — motor de geração (Groq/fallback Ollama), servidor Ollama e nº de documentos na vectorstore.
- 🌙 **Comandos de saída** — digitar `sair`, `quit` ou `exit` encerra a conversa educadamente.

### Como executar

```bash
# A partir da raiz do projeto
python interface/app.py

# Ou com porta personalizada / link público temporário
python interface/app.py --port 7860 --share
```

Acesse no navegador em `http://127.0.0.1:7860`.

> 💡 A interface também pode ser iniciada pelo menu do CLI (**opção 7**).

---

## 🧠 Agente de IA (tool-calling)

A **Fase G1** transforma o Ynuyasha em um **agente ReAct** com *tool-calling* quando
há uma chave Groq configurada: o modelo decide qual ferramenta consultar, executa a
busca e responde com base no resultado — sem passar pelos embeddings do Ollama, o que
deixa perguntas estruturadas (ex.: *"o que é um parsec?"*) quase instantâneas.

### Arquitetura

```
┌──────────────────────────────────────────────────────────────────────┐
│  agente_ia.py — StateGraph(MessagesState)  recursion_limit=6        │
│                                                                      │
│   agente (ChatGroq.bind_tools)  →  tools_condition  →  ferramentas   │
│        ▲                                          │   (ToolNode)      │
│        └────────────────  loop  ◀─────────────────┘                  │
└──────────────────────────────────────────────────────────────────────┘
        │                                      │
        ▼                                      ▼
  sem Groq → degrada para RAG clássico   ferramentas.py (9 tools)
  (preparar_contexto + gerar_resposta)   leem os CSVs diretamente
```

### As 9 ferramentas (`src/tratamento/ferramentas.py`)

| Ferramenta | Dataset (CSV) | Pesquisa por |
|---|---|---|
| `consultar_planetas` | `planetas_e_estrelas_rag.csv` | nome do planeta/estrela |
| `consultar_habitabilidade` | `habitabilidade_exoplanetas.csv` | nome do planeta |
| `consultar_asteroides` | `asteroides_cometas_jpl.csv` | nome do corpo |
| `consultar_constelacao` | `constelacoes_iau.csv` | nome/sigla da constelação |
| `consultar_glossario` | `glossario_astronomico_conceitos.csv` | termo científico |
| `consultar_objeto_simbad` | `estrelas_e_objetos_simbad.csv` | identificador (M 31...) |
| `consultar_estrelas_gaia` | `estrelas_proximas_gaia.csv` | id Gaia DR3 |
| `consultar_eventos` | `eventos_transientes_extremos.csv` | identificador/tipo (QSO...) |
| `buscar_na_base` | RAG (recuperar_contexto) | pergunta livre em toda a base |

Cada consulta devolve até 5 linhas no formato texto legível + **`Fonte: <arquivo>, Linha X`**,
que o agente preserva nas citações (mesma convenção da RAG).

### Degradação segura (sem Groq)

- Sem `GROQ_API_KEY` (ou com o placeholder `sua_chave_aqui`), `agente.py:responder()`
  mantém o fluxo RAG clássico (recuperação → geração) — comportamento inalterado.
- O grafo é compilado **uma única vez** e cacheado (singleton), e o LLM com ferramentas
  vinculadas também é reutilizado entre chamadas.

### Interface web (Gradio)

Com Groq disponível, o chat da interface também passa pelo agente ReAct (resposta rápida
por ferramentas); sem Groq, mantém o streaming RAG com o painel de contexto.

### Uso

```bash
# Terminal (rota automática: Groq → agente; senão → RAG)
python -m src.tratamento.agente "Qual a definição de parsec?"
python -m src.tratamento.agente "Quais asteroides são potencialmente perigosos?"

# Painel de status mostra o modo ativo:
#   "Agente ReAct (9 ferramentas)" (com Groq) ou "RAG simples" (sem Groq)
```

---

## 📊 Avaliação e métricas

A **Fase H** adiciona uma avaliação quantitativa do agente, com um *benchmark*
de perguntas em pt-BR e um gerador de relatório em Markdown.

### Benchmark (`data/avaliacao/benchmark.json`)

- **~30 perguntas** cobrindo os **8 datasets** (planetas, habitabilidade,
  asteroides, constelações, glossário, SIMBAD, Gaia e eventos) com
  `linhas_esperadas` — os índices (0-based) das linhas dos CSVs que a resposta
  correta deve citar.
- **Casos "fora da base"** — perguntas cuja resposta correta é recusar
  educadamente (teste de honestidade).

### Camadas avaliadas (`src/tratamento/avaliacao.py`)

| Camada | Como | Métricas |
|---|---|---|
| **Ferramentas (offline)** | `.invoke` real nos CSVs, conferindo as citações `Fonte: <arquivo>, Linha X` | recall, precisão, MRR, nDCG, hit@1 |
| **Retrieval (RAG)** | `recuperar_contexto` real, comparando metadados `(source, row)` | recall@k, precisão@k, MRR, nDCG@k, hit@1 |
| **Agente (--online)** | `executar_agente` via Groq, validando citação/substring | citação correta / recusa honesta |

As métricas são funções puras sobre conjuntos de chaves `(source, row)`, o que
permite testes herméticos com fakes.

### Como rodar

```bash
# Offline (ferramentas + fora da base; a seção de retrieval é pulada sem a
# base vetorial/Ollama disponível)
python -m src.tratamento.avaliacao

# Online — inclui a camada de agente via Groq (requer GROQ_API_KEY)
python -m src.tratamento.avaliacao --online

# Testes automatizados da avaliação (herméticos, sem Ollama)
python -m pytest testes/test_avaliacao.py -q
```

> ⚠️ A seção **Retrieval** usa embeddings do Ollama; se você não quiser
> sobrecarregar a máquina, use apenas o modo offline ou os testes.

O relatório é salvo em `data/avaliacao/resultados/resultado_<timestamp>.md`
(facilita comparar a evolução entre fases).

---

## 🧠 Memória persistente

O agente mantém **histórico entre turnos** graças a um checkpointer do
LangGraph (`src/tratamento/memoria.py`):

- **Persistência real:** `SqliteSaver` em `data/checkpoints/conversas.sqlite`
  (sobrevive a reinícios do processo).
- **Fallback:** `MemorySaver` em memória quando o pacote sqlite não está
  disponível (ex.: testes herméticos).
- `novo_thread_id()` gera identificadores únicos de sessão.

### Como funciona

- `agente_ia.executar_agente(..., thread_id=...)` compila o grafo com o
  checkpointer e **ancora o estado pelo `thread_id`**: o `SystemMessage` é
  injetado apenas quando o estado da thread está vazio (via `get_state`).
- **Sem `thread_id`**, o comportamento anterior é mantido (System + histórico
  explícito, sem persistência).
- **CLI** (`src/tratamento/agente.py`): uma thread por sessão; o comando
  `nova conversa` reinicia a memória.
- **Interface web** (`interface/app.py`): uma thread por janela; o botão
  "Limpar conversa" reinicia a sessão.

```bash
# Testes da memória (MemorySaver + FakeLLM, sem Ollama/Groq)
python -m pytest testes/test_memoria.py -q
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

**Fase G1 concluída** — agente ReAct com tool-calling (Groq):
- 9 ferramentas que leem os CSVs diretamente (respostas rápidas) ✔
- Grafo LangGraph com loop agente → ferramentas (recursion_limit=6) ✔
- Degradação segura para RAG sem Groq ✔
- Status "Agente ReAct (9 ferramentas)" / "RAG simples" no painel ✔
- Interface web roteada pelo agente quando a Groq está disponível ✔

**Fase H concluída** — avaliação quantitativa + memória persistente:
- Benchmark com ~30 perguntas cobrindo os 8 datasets + casos fora da base ✔
- Métricas (recall, precisão, MRR, nDCG, hit@1) e relatório Markdown ✔
- Avaliação offline (ferramentas) e opcional --online (agente via Groq) ✔
- Checkpointer SQLite (data/checkpoints/conversas.sqlite) + fallback MemorySaver ✔
- Thread por sessão no CLI e por janela na interface; "nova conversa" reinicia ✔

**Fase I concluída** — robustez e observabilidade (mantendo o agente simples):
- Poda de memória (últimas N mensagens ao LLM; System preservado) ✔
- Fallback automático: se a ferramenta específica não achar, tenta `buscar_na_base` ✔
- Fallback automático por erro da Groq (quota 429, `tool_use_failed`, rede) → RAG/Ollama local ✔
- **Gate de ancoragem**: respostas sem dados/citação da base são substituídas pela recusa (sem conhecimento próprio do modelo); citações no plural são aceitas e citações forjadas são bloqueadas ✔
- Log por turno em CSV (desligado por padrão; `YNUVASHA_LOG_TURNOS=true`) ✔
- Feedback 👍/👎 na interface web → `data/avaliacao/feedback.csv` ✔
- Robustez de embeddings (timeout/keep-alive, cache de query, warm-up no boot) + lock anti-rebuild concorrente ✔

---

## 🧪 Fase I — robustez e observabilidade

A **Fase I** deixa o agente mais robusto e observável, sem complicar a arquitetura.

### Poda de memória (contexto)
Em conversas longas, apenas as últimas `LIMITE_MENSAGENS_AGENTE` (12) mensagens são
enviadas ao LLM a cada invocação (`agente_ia._ultimas_mensagens`), preservando sempre
o `SystemMessage`. O checkpointer continua guardando o histórico completo no sqlite.

### Fallback automático (não recusar cedo demais)
O `PROMPT_AGENTE` agora instrui: se uma ferramenta específica retornar
"Nenhum registro…", chame `buscar_na_base` com a pergunta original antes de
concluir que a informação não existe.

### Gate de ancoragem (não responder do conhecimento próprio)
A recusa de perguntas fora da base é **imposta em código**, não só por instrução:

- **RAG (`geraçao.py`):** contexto vazio → o LLM nem é consultado; `gerar_resposta`
  e `gerar_resposta_stream` devolvem `MENSAGEM_FORA_DA_BASE`.
- **Agente ReAct (`agente_ia.py`):** após o `grafo.invoke`, `_verificar_ancoragem`
  aceita a resposta se ela tiver **pelo menos uma** citação `Fonte: <arquivo>,
  Linha X` presente no retorno real das ferramentas (retorno tolera o formato RAG
  `fonte: <arquivo>, linha: <X>` de `buscar_na_base`). Citações no plural — ex.:
  `Fonte: X, Linhas 5, 7 e 10` — são interpretadas corretamente pelo parser
  (`_extrair_citacoes_resposta`). Se **qualquer** citação for forjada (não vier das
  ferramentas), a resposta é substituída pela recusa; respostas sem citação (com
  dados disponíveis) ou sem nenhum dado de ferramenta também são barradas. Com
  `thread_id`, o gate considera apenas as ferramentas do turno atual.

> A regra foi relaxada para **não transformar respostas corretas em recusa**: antes
> era exigido que todas as citações da resposta existissem no retorno das ferramentas,
> o que descartava respostas válidas (ex.: Groq agrupando linhas no plural). Agora
> basta uma citação válida e **nenhuma** forjada — o bloqueio a fontes inventadas é
> mantido.

Isso garante o comportamento documentado no início do README: responder **apenas**
com base na base de conhecimento, mesmo quando o modelo "saberia" a resposta.

### Log por turno (desligado por padrão)
Grava cada turno (modo, ferramentas chamadas, citação, latência, pergunta e
resposta) em `data/avaliacao/turnos_log.csv`. **Desligado por padrão**:

```env
# .env — ative para registrar cada turno
YNUVASHA_LOG_TURNOS=true
```

Use `false` (ou deixe sem a variável) para manter o comportamento silencioso.

### Feedback na interface web
Ao lado do chat há botões **👍/👎** que gravam a avaliação da última resposta em
`data/avaliacao/feedback.csv` — material para futuras rodadas de avaliação.

Os arquivos `turnos_log.csv` e `feedback.csv` não são versionados (dados de uso);
os relatórios de avaliação continuam em `data/avaliacao/resultados/`.
