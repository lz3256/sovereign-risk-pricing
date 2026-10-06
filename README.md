# Sovereign Risk Pricing

Research code testing whether differences between sovereign market pricing and fundamental risk measures add predictive information beyond existing market signals. CDS and dollar-bond outcomes are evaluated separately.

## What is implemented

- Monthly information-state assembly with quote-age and quality checks.
- Provider risk-premium scores and a fixed three-component comparison.
- Paired cross-country predictions on common samples.
- Annual expanding-window ridge models with training-only standardization and unfinished-label exclusion.
- Synchronized month-block resampling, simultaneous primary intervals, and predeclared sensitivity comparisons.

## Run without data access

Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py
python -m unittest discover -s tests -v
```

Open `results/demo/report.html`. The demo constructs an artificial monthly feature panel and exercises the chronological prediction/paired-metric stage. It does not simulate provider score construction or validate the daily source adapter.

For an authorized QDF file, use `python sovereign_value.py --input /path/to/authorized-qdf.csv`. That entry point assembles monthly states, estimates the full specification, and writes reports locally. Without an explicit file it uses matching local cache or your own locally configured DataQuery credentials; the demo never reads credentials.

## Selected preliminary findings

The recorded primary 63-business-day comparison did not establish stable forecast improvement over the market baseline. Both the provider score and fixed-component comparison had primary simultaneous intervals containing zero. Sensitivity checks did not establish a robust improvement either.

This is a concise account of recorded exploratory evidence, not a new empirical run. Detailed estimates, historical screening lists, source panels, individual predictions, and empirical charts are withheld. The finding does not establish that all fundamental pricing information lacks value. See [method and interpretation](docs/METHOD.md).

## Data sources and boundaries

The empirical implementation uses JPMaQS sovereign macro/risk-premium scores, CDS, and dollar-bond series through the [DataQuery interface](https://docs.macrosynergy.com/latest/macrosynergy.download.jpmaqs.html). Service names identify data inputs rather than institutional affiliation. No vendor observations or credentials are included; users need their own authorized access.

Provider-defined constituent and model vintages have not been independently verified. Time-ordered validation therefore does not certify a fully point-in-time trading backtest. Bond outcomes retain interest-rate/duration exposure, and the project does not estimate executable strategy returns after costs.

## Public package verification

The prepared public copy passed 7 existing automated checks and its credential-free synthetic demonstration. No real-data download or empirical rerun was performed for this verification. Runtime details are in [VALIDATION.json](VALIDATION.json).

## Credits

Project owner: Linfeng Zhao (@lz3256). Developed with substantial OpenAI Codex assistance; see [credits](CREDITS.md).
