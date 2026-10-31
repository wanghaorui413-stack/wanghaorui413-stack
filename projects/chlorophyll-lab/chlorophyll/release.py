import shutil
import joblib
from pathlib import Path
from .common import atomic_directory, read_json, write_json, sha256, named_path


def export_model(run, destination):
    source = named_path("runs", run)
    card = read_json(source / "model_card.json")
    selected = card["selected_model"]
    with atomic_directory(Path(destination)) as stage:
        shutil.copy2(source / "preprocess.npz", stage / "preprocess.npz")
        files = ["preprocess.npz"]
        if selected in ["mlp", "cnn"]:
            shutil.copy2(source / f"{selected}.pt", stage / f"{selected}.pt")
            files.append(f"{selected}.pt")
        elif selected != "mean_log":
            baseline = joblib.load(source / "baselines.joblib")
            joblib.dump({selected: baseline[selected]}, stage / "baselines.joblib")
            files.append("baselines.joblib")
        card["release"] = {"source_model_card_sha256": sha256(source / "model_card.json"), "model": selected}
        card["available_models"] = [selected]
        card["artifact_hashes"] = {name: sha256(stage / name) for name in files}
        write_json(stage / "model_card.json", card)
    return Path(destination)
