# Mehurcki

Zvočni detektor mehurčkov

Projekt za predmet Globoko učenje na UL FRI.

## Predpriprava

```bash
uv sync
```

# Uporaba

```bash
python -m src.main train_detector --name constant --preprocessor identity
python -m src.main eval_detector --model_file models/constant_model --file data/test.wav
```
