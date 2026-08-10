# 🐕 Ynuyasha — Agente RAG de Astronomia

> **Ynuyasha** é um agente de inteligência artificial baseado em **RAG (Retrieval-Augmented Generation)** especializado em **astronomia**. Ele consulta uma base de conhecimento construída a partir de fontes científicas reais e responde em **português do Brasil**, de forma clara, objetiva e com **fontes citadas**.

---

## 📋 Descrição Geral

O Ynuyasha é um assistente que roda **no terminal** e responde perguntas sobre astronomia usando apenas os dados presentes na sua base de conhecimento. Ele não inventa números ou fatos: se a informação não estiver nos dados, ele diz que não encontrou.

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
├── data/
│   ├── dataset/                 # CSVs gerados pelas consultas
│   ├── documentos/              # Arquivos de apoio ao pipeline
│   └── vectorstore/             # Embeddings persistidos (JSON)
├── src/
│   ├── consultas/               # Scripts de coleta de dados (API)
│   └── tratamento/
│       ├── loading.py           # Carrega e divide documentos
│       ├── embeddings.py        # Modelo de embedding (Ollama)
│       ├── base_vetorial.py     # Cria/persiste a vectorstore
│       ├── retrieval.py         # Busca semântica por similaridade
│       ├── geração.py           # Geração de resposta (Groq/Ollama)
│       └── agente.py            # Pipeline completo (retrieval → geração)
└── testes/                      # Scripts de checagem de dados
```

### Camadas do pipeline

| Camada | Módulo | Função |
|---|---|---|
| **Coleta** | `src/consultas/` | Busca dados reais nas APIs astronômicas e gera CSVs |
| **Ingestão** | `loading.py` | Carrega os CSVs e divide em chunks (1000 caracteres, overlap 100) |
| **Embeddings** | `embeddings.py` | Converte textos em vetores com `nomic-embed-text` (Ollama) |
| **Indexação** | `base_vetorial.py` | Monta e persiste a vectorstore em JSON (190 chunks, com checkpoint) |
| **Retrieval** | `retrieval.py` | Busca os 5 chunks mais relevantes por similaridade vetorial |
| **Geração** | `geração.py` | Gera a resposta final em Markdown, citando fontes e linhas |

---

## 🛠️ Tecnologias e Ferramentas

| Tecnologia | Uso |
|---|---|
| **Python 3.14** | Linguagem principal |
| **LangChain** | Orquestração do pipeline RAG (core, community, splitter) |
| **Ollama** | Embeddings (`nomic-embed-text`) e fallback de geração |
| **Groq** | Geração principal (`llama-3.3-70b-versatile`) |
| **Rich** | Interface visual no terminal (tabelas, painéis, Markdown) |
| **Questionary** | Menus e prompts interativos |
| **Pyfiglet** | Banner ASCII do Ynuyasha |
| **pandas / requests** | Processamento de dados e chamadas HTTP |

### Dependências de runtime

- **Ollama** rodando localmente com o modelo `nomic-embed-text`:
  ```bash
  ollama pull nomic-embed-text
  ```
- **Chave Groq** (opcional — sem ela, o agente usa o fallback local do Ollama):
  preencher `GROQ_API_KEY` no arquivo `.env`.

---

## 🚀 Como Executar

### Pré-requisitos

1. **Python 3.10+** instalado.
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
GROQ_API_KEY=sua_chave_aqui
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.3

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
| 0 | Sair da aplicação |

### Comandos diretos

```bash
# Responder uma pergunta única pelo terminal
python -m src.tratamento.agente "O que é um parsec?"

# Ver o contexto recuperado para uma pergunta (sem gerar resposta)
python -m src.tratamento.retrieval "planeta em zona habitável"

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
> Fonte: `data/dataset/habitabilidade_exoplanetas.csv`, linha 26

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
