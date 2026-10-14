# Italian Tech Salaries Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Dataset: HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-datapizza--ai--lab%2Fsalaries-yellow)](https://huggingface.co/datasets/datapizza-ai-lab/salaries)
[![Dataset revision](https://img.shields.io/badge/dataset%20revision-52ad2e7%20(2026--10--11)-yellow)](https://huggingface.co/datasets/datapizza-ai-lab/salaries/tree/52ad2e7107a0c51d224ff272475b7b650daabc88)
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

- **Esperienza** e il fattore dominante: un senior guadagna oltre il 50% in piu di un junior (Cohen's d = 0.96). La regressione stima +3.2% per ogni anno in media, con rendimenti decrescenti: circa +7% a inizio carriera, quasi zero intorno ai 18 anni
- **Ruolo**: secondo fattore per importanza. Aggiungerlo alla regressione porta l'R² da 0.25 a 0.37
- **Divario Nord-Sud** confermato: gap di ~6.6k tra Nord e Sud. In euro cresce con la seniority, ma in percentuale resta stabile (~17-19% a ogni livello). Il Centro si posiziona a meta strada
- **Gender pay gap**: 11.5% grezzo, circa 7.4% a parita di esperienza, ruolo, regione, dimensione aziendale e work mode. Persiste a ogni livello di esperienza anche dopo la correzione per test multipli. La quota femminile scende dal 16.5% tra i junior al 9.9% tra i senior, un pattern coerente con un *leaky pipeline* o con un effetto coorte
- **Importanza dei fattori** (eta² parziale): esperienza > ruolo > work mode > regione > gender > dimensione azienda. A parita di ruolo, il lavoro onsite e associato a un -13% rispetto all'hybrid e il Sud a un -8% rispetto al Centro; il remote a un +2.8%

## Metodi Statistici Utilizzati

| Test | Scopo |
|------|-------|
| **ANOVA / Welch ANOVA / Kruskal-Wallis** | Confronto tra 3+ gruppi (parametrico / varianze diverse / non parametrico) |
| **Welch t-test** | Confronto tra 2 gruppi senza assumere varianze uguali |
| **Games-Howell** | Post-hoc per identificare quali coppie differiscono, senza assumere varianze uguali |
| **Correzione di Holm** | Controllo dei falsi positivi quando si eseguono piu test sulla stessa domanda |
| **Cohen's d** | Dimensione dell'effetto (rilevanza pratica) |
| **Eta² parziale** | Importanza relativa dei fattori di un modello, sulla stessa scala |
| **Chi Quadro** | Indipendenza tra variabili categoriche |
| **Regressione OLS su log(salario)** | Effetto netto percentuale di ciascun fattore, controllando per gli altri (errori standard robusti HC3) |
| **Shapiro-Wilk / Levene** | Verifica assunzioni ANOVA (normalita, omoschedasticita) |

## Riproducibilita

Il dataset viene aggiornato ogni settimana su HuggingFace, quindi caricando sempre l'ultima versione i numeri cambierebbero nel tempo. Per questo `load_salaries()` carica una **revisione fissata** del dataset, e tutti i risultati del notebook e di questo README si riferiscono a quella:

| | |
|---|---|
| **Revisione** | [`52ad2e7`](https://huggingface.co/datasets/datapizza-ai-lab/salaries/tree/52ad2e7107a0c51d224ff272475b7b650daabc88) |
| **Export** | 2026-10-11 (weekly export) |
| **Risposte** | 26.478 |
| **Definita in** | `DATASET_REVISION` in [`src/salaries_analysis/data/dataset.py`](src/salaries_analysis/data/dataset.py) |

Per aggiornare l'analisi a una versione piu recente:

1. Scegliere un commit dalla [cronologia del dataset](https://huggingface.co/datasets/datapizza-ai-lab/salaries/commits/main)
2. Aggiornare `DATASET_REVISION` (e il commento con data e numero di risposte)
3. Ri-eseguire il notebook e aggiornare i numeri citati nei testi, nel README e la tabella qui sopra

Per un'esplorazione veloce sull'ultima versione disponibile, senza toccare la revisione fissata: `load_salaries(revision=None)`.

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
│   │   ├── effect_size.py           # Cohen's d, eta² parziale
│   │   ├── results.py               # Dataclass per risultati (ANOVA, Chi2, TTest, AssumptionResult)
│   │   └── tests.py                 # Welch t-test, (Welch) ANOVA, Kruskal-Wallis, Games-Howell, Chi2, Holm
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

Aprire `data_analysis.ipynb` in Jupyter o VS Code. Il dataset viene scaricato automaticamente da HuggingFace al primo avvio, nella revisione fissata (vedi [Riproducibilita](#riproducibilita)).

### Test

```bash
uv run pytest tests/ -v
```
