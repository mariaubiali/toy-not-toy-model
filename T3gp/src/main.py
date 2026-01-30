from __future__ import annotations
import os
import sys
import yaml
import shutil

from networks.gp_nuts import run_from_config as run_gp
from networks.neuralnet import run_from_config as run_nn

def main():
    cfg_path = os.environ.get("T3GP_CONFIG", "configs/configs.yaml")
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    out = cfg.get("output_dir", "outputs/run")
    os.makedirs(out, exist_ok=True)

    dst = os.path.join(out, "config.yaml")
    shutil.copy(cfg_path, dst)

    print(f"Saved config to {dst}")

    model_type = str(cfg.get("model", {}).get("type", "gp")).lower()
    if model_type == "gp":
        run_gp(cfg)
    elif model_type == "nn":
        run_nn(cfg)
    else:
        raise ValueError(f"Unknown model.type={model_type!r}")

if __name__ == "__main__":
    main()
