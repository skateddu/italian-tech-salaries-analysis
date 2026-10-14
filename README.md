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

- **Esperienza** e il fattore dominante: un senior guadagna oltre il 50% in piu di un junior (Cohen's d = 0.96); la regressione stima +3.3% per ogni anno aggiuntivo
- **Divario Nord-Sud** confermato: gap di ~6.6k tra Nord e Sud. In euro cresce con la seniority, ma in percentuale resta stabile (~17-19% a ogni livello). Il Centro si posiziona a meta strada
- **Gender pay gap**: 11.5% grezzo, circa 7.9% a parita di esperienza, regione, dimensione aziendale e work mode. Persiste a ogni livello di esperienza anche dopo la correzione per test multipli. A questo si aggiunge il fenomeno del *leaky pipeline* (quota femminile dal 16.5% tra i junior al 9.9% tra i senior)
- **Regressione multipla su log(salario)**: oltre all'esperienza, le penalita maggiori sono associate al lavoro onsite (-13.5% rispetto all'hybrid) e al Sud (-10% rispetto al Centro); gender e dimensione aziendale valgono circa l'8%. Remote e hybrid sono equivalenti

## Metodi Statistici Utilizzati

| Test | Scopo |
|------|-------|
| **ANOVA / Welch ANOVA / Kruskal-Wallis** | Confronto tra 3+ gruppi (parametrico / varianze diverse / non parametrico) |
| **Welch t-test** | Confronto tra 2 gruppi senza assumere varianze uguali |
| **Tukey HSD** | Post-hoc per identificare quali coppie differiscono |
| **Correzione di Holm** | Controllo dei falsi positivi quando si eseguono piu test sulla stessa domanda |
| **Cohen's d** | Dimensione dell'effetto (rilevanza pratica) |
| **Chi Quadro** | Indipendenza tra variabili categoriche |
| **Regressione OLS su log(salario)** | Effetto netto percentuale di ciascun fattore, controllando per gli altri |
| **Shapiro-Wilk / Levene** | Verifica assunzioni ANOVA (normalita, omoschedasticita) |

## Riproducibilita

Il dataset viene aggiornato ogni settimana su HuggingFace. Per rendere i risultati riproducibili, `load_salaries()` carica una revisione fissata (`DATASET_REVISION` in `src/salaries_analysis/data/dataset.py`, export del 2026-10-11 con 26.478 risposte). Per aggiornare l'analisi basta cambiare la revisione e ri-eseguire il notebook; `load_salaries(revision=None)` carica l'ultima versione disponibile.

## Struttura del Progetto

```text
italian-tech-salaries-analysis/
├── data_analysis.ipynb              # Notebook principale dell'analisi
├── src/salaries_analysis/
│   ├── config.mplstyle              # Stile Matplotlib personalizzato
│   ├── data/
│   │   ├── dataset.py               # Caricamento dataset da HuggingFace (revisione fissata)
│   │   ├── descriptive.py           # Statistiche descrittive (describe + skew/kurtosis)
│   │   ├── features.py              # Feature engineering
│   │   └── geo_mapper.py            # Mappatura province -> regioni -> macro-regioni
│   ├── statistics/
│   │   ├── effect_size.py           # Cohen's d
│   │   ├── results.py               # Dataclass per risultati (ANOVA, Chi2, TTest, AssumptionResult)
│   │   └── tests.py                 # Welch t-test, (Welch) ANOVA, Kruskal-Wallis, Chi2, assunzioni, Holm
│   └── visualization/
│       └── plots.py                 # Barplot, boxplot, heatmap
├── tests/
│   ├── conftest.py                  # Fixture condivise (DataFrame, gruppi numerici)
│   ├── test_dataset.py              # Test caricamento dataset
│   ├── test_descriptive.py          # Test statistiche descrittive
│   ├── test_effect_size.py          # Test Cohen's d
│   ├── test_features.py             # Test feature engineering
│   ├── test_geo_mapper.py           # Test mappatura geografica
│   ├── test_results.py              # Test dataclass risultati
│   └── test_tests.py                # Test wrapper test statistici
├── .github/workflows/ci.yml         # CI: ruff + pytest su Python 3.10-3.13
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
| [statsmodels](https://www.statsmodels.org/) | Regressione OLS, ANOVA fattoriale, Welch ANOVA e correzione di Holm | [Docs](https://www.statsmodels.org/stable/index.html) |
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
