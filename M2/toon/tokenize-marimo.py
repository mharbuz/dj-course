import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Zerknij do pliku `HOMEWORK.md` aby zobaczyć opis zadania domowego :)
    """)
    return


@app.cell
def _():
    from tokenizers import Tokenizer
    import json
    import os

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TOKENIZER_DIR = os.path.join(SCRIPT_DIR, 'tokenizers')
    SAMPLES_DIR = os.path.join(SCRIPT_DIR, 'samples')
    return SAMPLES_DIR, TOKENIZER_DIR, Tokenizer, json, os


@app.cell
def _(TOKENIZER_DIR, Tokenizer, os):
    TOKENIZER_NAME = "qwen3-4b-tokenizer"
    ALL_TOKENIZERS = {}

    if not os.path.isdir(TOKENIZER_DIR):
        print(f"❌ Error: Tokenizer directory not found at {TOKENIZER_DIR}")
        exit(1)

    for tokenizer_fname in os.listdir(TOKENIZER_DIR):
        if tokenizer_fname.endswith('.json'):
            key = tokenizer_fname[:-5]  # remove .json
            full_path = os.path.join(TOKENIZER_DIR, tokenizer_fname)
            try:
                ALL_TOKENIZERS[key] = Tokenizer.from_file(full_path)
            except Exception as e:
                print(f"❌ Error loading tokenizer '{key}' from '{full_path}': {e}")


    if TOKENIZER_NAME not in ALL_TOKENIZERS:
        print(f"❌ Error: Tokenizer '{TOKENIZER_NAME}' not found in {TOKENIZER_DIR}")
        exit(1)

    tokenizer = ALL_TOKENIZERS[TOKENIZER_NAME]
    print(f"✅ Successfully loaded tokenizer: {TOKENIZER_NAME}")
    return TOKENIZER_NAME, tokenizer


@app.cell
def _(SAMPLES_DIR, os):
    # Lista próbek: bazowe nazwy z plików *-nows.json w samples/
    sample_base_names = []
    if os.path.isdir(SAMPLES_DIR):
        for sample_fname in sorted(os.listdir(SAMPLES_DIR)):
            if sample_fname.endswith("-nows.json"):
                base = sample_fname[:-10]  # remove -nows.json
                sample_base_names.append(base)
    return sample_base_names


@app.cell
def _(SAMPLES_DIR, os, sample_base_names, tokenizer):
    BAR_LEN = 20
    FORMAT_LABELS = {
        "json": "JSON",
        "nows-json": "JSON compact",
        "toon": "TOON",
        "yaml": "YAML",
    }
    EXTENSIONS = {"json": ".json", "nows-json": "-nows.json", "toon": ".toon", "yaml": ".yaml"}

    results = {}

    for SAMPLE_NAME in sample_base_names:
        sample_data = {}
        for fmt, ext in EXTENSIONS.items():
            path = os.path.join(SAMPLES_DIR, f"{SAMPLE_NAME}{ext}")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    sample_data[fmt] = f.read()
            except FileNotFoundError:
                sample_data[fmt] = ""

        if all(value == "" for value in sample_data.values()):
            continue
        try:
            counts = {}
            for fmt in ("json", "nows-json", "toon", "yaml"):
                text = sample_data.get(fmt, "") or ""
                counts[fmt] = len(tokenizer.encode(text).ids)
            results[SAMPLE_NAME] = counts
        except Exception as e:
            print(f"❌ Error processing '{SAMPLE_NAME}': {e}")

    return BAR_LEN, EXTENSIONS, FORMAT_LABELS, results


@app.cell
def _(BAR_LEN, FORMAT_LABELS, results, mo):
    def make_bar(ratio):
        filled = round(ratio * BAR_LEN)
        return "█" * filled + "░" * (BAR_LEN - filled)

    lines = []
    for sample_name in sorted(results.keys()):
        sample_counts = results[sample_name]
        min_tokens = min(sample_counts.values())
        sorted_formats = sorted(sample_counts.keys(), key=lambda k: sample_counts[k])
        lines.append(sample_name)
        for i, format_key in enumerate(sorted_formats):
            n = sample_counts[format_key]
            pct = 100.0 * min_tokens / n if n else 0
            bar = make_bar(min_tokens / n) if n else make_bar(0)
            label = FORMAT_LABELS[format_key]
            prefix = "→ " if i == 0 else "  "
            lines.append(f"{prefix}{label:<14} {bar}    {pct:5.1f}% ({n})")
        lines.append("")

    chart_text = "\n".join(lines)
    mo.md(f"```\n{chart_text}\n```")
    return chart_text, make_bar


if __name__ == "__main__":
    app.run()
