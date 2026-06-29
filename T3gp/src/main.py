from __future__ import annotations
import os
import sys
import yaml

from networks.gp_nuts import run_from_config as run_gp
from networks.neuralnet import run_from_config as run_nn


def parse_override_value(value: str):
    """
    Parse command-line override values using YAML syntax.

    Examples:
      "100"      -> int
      "4.0"      -> float
      "true"     -> bool
      "false"    -> bool
      "null"     -> None
      "[32, 32]" -> list
    """
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def set_nested_value(cfg: dict, dotted_key: str, value):
    """
    Set a nested config value using dot notation.

    Example:
      nn.ensemble.n_members=50

    modifies:
      cfg["nn"]["ensemble"]["n_members"] = 50
    """
    keys = dotted_key.split(".")
    target = cfg

    for key in keys[:-1]:
        if key not in target or target[key] is None:
            target[key] = {}
        target = target[key]

    target[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str]):
    """
    Apply command-line overrides of the form:

      key=value

    where key may use dot notation, e.g.

      output_dir=results/test
      nn.ensemble.n_members=50
      nn.repulsive.beta=4.0
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override {override!r}. Expected format key=value."
            )

        key, value = override.split("=", 1)
        set_nested_value(cfg, key, parse_override_value(value))


def main():
    cfg_path = os.environ.get("T3GP_CONFIG", "configs/configs.yaml")

    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Everything after the config path is treated as an override:
    #
    #   python src/main.py configs/config_re.yaml nn.ensemble.n_members=50
    #
    overrides = sys.argv[2:]
    apply_overrides(cfg, overrides)

    out = cfg.get("output_dir", "outputs/run")
    os.makedirs(out, exist_ok=True)

    dst = os.path.join(out, "config.yaml")

    # Save the resolved config, including command-line overrides.
    # This is better than shutil.copy, because the copied config would not
    # include the changed n_members/output_dir values.
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Saved config to {dst}")

    model_type = str(cfg.get("model", {}).get("type", "gp")).lower()
    if model_type == "gp":
        run_gp(cfg)
    elif model_type in ("nn", "ntk"):
        run_nn(cfg)
    else:
        raise ValueError(f"Unknown model.type={model_type!r}")


if __name__ == "__main__":
    main()