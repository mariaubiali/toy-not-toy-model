from pathlib import Path
import yaml
import yadism
from yadbox.export import dump_pineappl_to_file  # adjust if needed

BASE = Path(__file__).resolve().parent.parent
THEORY_PATH = BASE / "data" / "theory_cards" / "208.yaml"
YAMLDB = BASE / "data" / "observable_cards"
GRIDDIR = BASE / "data" / "grids" / "208"

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_obs_key(obs_card):
    return list(obs_card["observables"].keys())[0]

def main():
    GRIDDIR.mkdir(parents=True, exist_ok=True)
    theory = load_yaml(THEORY_PATH)

    yaml_files = sorted(YAMLDB.rglob("*.yaml"))
    for yml in yaml_files:
        obs = load_yaml(yml)
        dataset_name = yml.stem
        obs_key = get_obs_key(obs)

        print(f"Running yadism for {yml}")
        out = yadism.run_yadism(theory, obs)

        outpath = GRIDDIR / f"{dataset_name}.pineappl.lz4"
        dump_pineappl_to_file(out, str(outpath), obs_key)

        print(f"wrote {outpath}")

if __name__ == "__main__":
    main()
