# Italian Tech Salaries Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Dataset: HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-datapizza--ai--lab%2Fsalaries-yellow)](https://huggingface.co/datasets/datapizza-ai-lab/salaries)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-notebook-F37626?logo=jupyter&logoColor=white)](data_analysis.ipynb)

> Analisi statistica dei salari nel settore tech italiano, basata sul dataset [datapizza-ai-lab/salaries](https://huggingface.co/datasets/datapizza-ai-lab/salaries) di HuggingFace.

---

## Panoramica

Il notebook esplora il mercato tech italiano rispondendo a quattro domande concrete, usando test statistici appropriati e verificando le assunzioni alla base di ciascuno:

1. **Quanto conta l'esperienza?** - Progressione salariale per seniority e impatto di ogni anno aggiuntivo
2. **Dove conviene lavorare?** - Divario Nord-Sud nel tech, stratificato per esperienza
3. **Esistono disuguaglianze strutturali?** - Gender pay gap, leaky pipeline e analisi a parita di esperienza
4. **Cosa fa davvero la differenza?** - Peso relativo di ogni fattore tramite regressione multipla

L'analisi segue uno schema consolidato per ogni sezione: *cosa si analizza*, *perche*, *quale test e come funziona*, *come leggere i risultati*, seguito dai dati e dall'interpretazione.

## Risultati Principali

- **Esperienza** e il fattore dominante: un senior guadagna oltre il 50% in piu di un junior (Cohen's d = 1.01), con un incremento significativo per ogni anno aggiuntivo
- **Divario Nord-Sud** confermato: gap di ~7k tra Nord e Sud, che si amplifica con la seniority. Il Centro si posiziona a meta strada
- **Gender pay gap** reale e strutturale: persiste a ogni livello di esperienza, anche controllando per regione e dimensione aziendale. A questo si aggiunge il fenomeno del *leaky pipeline* (quota femminile in calo ai livelli senior)
- **Gerarchia dei fattori** (regressione multipla): esperienza > regione > gender > dimensione azienda > work mode. Dimensione aziendale e work mode, spesso sopravvalutati, hanno un impatto modesto

## Metodi Statistici Utilizzati

| Test | Scopo |
|------|-------|
| **ANOVA / Kruskal-Wallis** | Confronto tra 3+ gruppi (parametrico / non parametrico) |
| **t-test** | Confronto tra 2 gruppi |
| **Tukey HSD** | Post-hoc per identificare quali coppie differiscono |
| **Cohen's d** | Dimensione dell'effetto (rilevanza pratica) |
| **Chi Quadro** | Indipendenza tra variabili categoriche |
| **Regressione OLS** | Effetto netto di ciascun fattore, controllando per gli altri |
| **Shapiro-Wilk / Levene** | Verifica assunzioni ANOVA (normalita, omoschedasticita) |

## Struttura del Progetto

```text
italian-tech-salaries-analysis/
├── data_analysis.ipynb          # Notebook principale dell'analisi
├── src/
│   ├── config.mplstyle          # Stile Matplotlib personalizzato
│   ├── data/
│   │   ├── dataset.py           # Caricamento dataset da HuggingFace
│   │   ├── descriptive.py       # Statistiche descrittive (describe + skew/kurtosis)
│   │   ├── features.py          # Feature engineering
│   │   └── geo_mapper.py        # Mappatura province -> regioni -> macro-regioni
│   ├── statistics/
│   │   ├── effect_size.py       # Cohen's d
│   │   ├── results.py           # Dataclass per risultati (ANOVA, Chi2, TTest, AssumptionResult)
│   │   └── tests.py             # t-test, ANOVA, Kruskal-Wallis, Chi2, verifica assunzioni
│   └── visualization/
│       └── plots.py             # Barplot, boxplot, heatmap
├── tests/
│   ├── conftest.py              # Fixture condivise (DataFrame, gruppi numerici)
│   ├── test_descriptive.py      # Test statistiche descrittive
│   ├── test_effect_size.py      # Test Cohen's d
│   ├── test_features.py         # Test feature engineering
│   ├── test_geo_mapper.py       # Test mappatura geografica
│   ├── test_results.py          # Test dataclass risultati
│   └── test_tests.py            # Test wrapper test statistici
├── pyproject.toml
├── uv.lock
└── LICENSE
```

## Dipendenze

| Package | Descrizione | Link |
|---------|-------------|------|
| [pandas](https://pandas.pydata.org/) | Manipolazione e analisi dati tabulari | [Docs](https://pandas.pydata.org/docs/) |
| [NumPy](https://numpy.org/) | Array numerici e operazioni matematiche | [Docs](https://numpy.org/doc/stable/) |
| [SciPy](https://scipy.org/) | Test statistici (t-test, ANOVA, Kruskal-Wallis, Chi Quadro) | [Docs](https://docs.scipy.org/doc/scipy/) |
| [statsmodels](https://www.statsmodels.org/) | Regressione OLS e ANOVA fattoriale | [Docs](https://www.statsmodels.org/stable/index.html) |
| [Matplotlib](https://matplotlib.org/) | Grafici e visualizzazione dati | [Docs](https://matplotlib.org/stable/contents.html) |
| [seaborn](https://seaborn.pydata.org/) | Visualizzazione statistica (boxplot, heatmap, histplot) | [Docs](https://seaborn.pydata.org/api.html) |
| [datasets](https://huggingface.co/docs/datasets/) | Caricamento dataset da HuggingFace Hub | [Docs](https://huggingface.co/docs/datasets/) |

## Quick Start

Requisiti: **Python >= 3.10**, [uv](https://docs.astral.sh/uv/)

### Installazione

```bash
git clone https://github.com/skateddu/italian-tech-salaries-analysis.git
cd italian-tech-salaries-analysis
uv sync
```

### Utilizzo

Aprire `data_analysis.ipynb` in Jupyter o VS Code. Il dataset viene scaricato automaticamente da HuggingFace al primo avvio.

### Test

```bash
uv run pytest tests/ -v
```
