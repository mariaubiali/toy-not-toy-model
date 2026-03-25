from pathlib import Path
import yaml
import copy

BASE = Path(__file__).resolve().parent.parent
COMMONDATA = BASE / "commondata"
YAMLDB = BASE / "data" / "observable_cards"

# One shared grid for both NC datasets
NC_XGRID = [
    1.0e-6, 2.0e-6, 5.0e-6,
    1.0e-5, 2.0e-5, 5.0e-5,
    1.0e-4, 2.0e-4, 5.0e-4,
    1.0e-3, 2.0e-3, 5.0e-3,
    1.0e-2, 2.0e-2, 5.0e-2,
    1.0e-1, 2.0e-1, 4.0e-1,
    6.0e-1, 8.0e-1, 1.0,
]

# You can keep CC the same or tune separately
CC_XGRID = [
    1.0e-6, 2.0e-6, 5.0e-6,
    1.0e-5, 2.0e-5, 5.0e-5,
    1.0e-4, 2.0e-4, 5.0e-4,
    1.0e-3, 2.0e-3, 5.0e-3,
    1.0e-2, 2.0e-2, 5.0e-2,
    1.0e-1, 2.0e-1, 4.0e-1,
    6.0e-1, 8.0e-1, 1.0,
]

BASE_TEMPLATE = {
    "prDIS": None,
    "ProjectileDIS": None,
    "TargetDIS": "proton",
    "PolarizationDIS": 0.0,
    "PropagatorCorrection": 0,
    "NCPositivityCharge": None,
    "interpolation_xgrid": None,
    "interpolation_is_log": True,
    "interpolation_polynomial_degree": 4,
    "observables": {},
}


def read_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def infer_projectile(name: str) -> str:
    n = name.upper()
    if "EM" in n or "ELECTRON" in n or "MINUS" in n:
        return "electron"
    if "EP" in n or "POSITRON" in n or "PLUS" in n:
        return "positron"
    return "electron"


def extract_mid(val):
    if isinstance(val, dict):
        if "mid" in val:
            return val["mid"]
        if "value" in val:
            return val["value"]
        if "min" in val and "max" in val:
            return 0.5 * (float(val["min"]) + float(val["max"]))
    return val


def yadism_obs_name(interaction: str, observable_name: str) -> str:
    if interaction == "CC":
        return "XSHERACC"
    if interaction == "NC":
        if "AVG" in observable_name.upper():
            return "XSHERANCAVG"
        return "XSHERANC"
    raise ValueError(f"Unknown interaction: {interaction}")


def xgrid_for_interaction(interaction: str):
    if interaction == "NC":
        return NC_XGRID
    if interaction == "CC":
        return CC_XGRID
    raise ValueError(f"Unknown interaction: {interaction}")


def build_cards(dataset_dir: Path, interaction: str):
    meta = read_yaml(dataset_dir / "metadata.yaml")
    setname = meta["setname"]

    for obs in meta["implemented_observables"]:
        observable_name = obs["observable_name"]
        full_name = f"{setname}_{observable_name}"

        kin_path = dataset_dir / obs["kinematics"]["file"]
        kin = read_yaml(kin_path)

        card = copy.deepcopy(BASE_TEMPLATE)
        card["prDIS"] = interaction
        card["ProjectileDIS"] = infer_projectile(observable_name)
        card["interpolation_xgrid"] = xgrid_for_interaction(interaction)

        points = []
        for b in kin["bins"]:
            x = float(extract_mid(b["x"]))
            q2 = float(extract_mid(b["Q2"]))
            y = float(extract_mid(b["y"]))
            points.append({"x": x, "Q2": q2, "y": y})

        internal_name = yadism_obs_name(interaction, observable_name)
        card["observables"] = {internal_name: points}

        YAMLDB.mkdir(parents=True, exist_ok=True)
        outpath = YAMLDB / f"{full_name}.yaml"

        with open(outpath, "w") as f:
            yaml.safe_dump(card, f, sort_keys=False)

        print(f"Wrote {outpath}")


def main():
    build_cards(COMMONDATA / "HERA_NC_318GEV", "NC")
    build_cards(COMMONDATA / "HERA_CC_318GEV", "CC")


if __name__ == "__main__":
    main()