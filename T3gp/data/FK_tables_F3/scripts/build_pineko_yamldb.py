from pathlib import Path
import yaml

BASE = Path(__file__).resolve().parent.parent
YAMLDB = BASE / "data" / "yamldb"

DATASETS = [
    "HERA_NC_318GEV_EM-SIGMARED",
    "HERA_NC_318GEV_EP-SIGMARED",
    "HERA_CC_318GEV_EM-SIGMARED",
    "HERA_CC_318GEV_EP-SIGMARED",
]

def main():
    YAMLDB.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        card = {
            "operation": None,
            "operands": [[ds]],
        }

        outpath = YAMLDB / f"{ds}.yaml"
        with open(outpath, "w") as f:
            yaml.safe_dump(card, f, sort_keys=False)

        print(f"Wrote {outpath}")

if __name__ == "__main__":
    main()