import json
import sys
from pathlib import Path

def clean_widgets(infile, outfile=None):
    infile = Path(infile)
    if outfile is None:
        outfile = infile.with_name(infile.stem + "_clean.ipynb")

    # Load notebook JSON
    with infile.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    # Remove widget metadata if present
    metadata = nb.get("metadata", {})
    if "widgets" in metadata:
        print("Removing widgets metadata...")
        metadata.pop("widgets")

    # Rewrite notebook without widgets
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Cleaned notebook saved to: {outfile}")

clean_widgets('...')
