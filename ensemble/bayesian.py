import json
from pathlib import Path

class ModelWeights:
    def __init__(self, path="ensemble/weights.json"):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {}  # {model_name: {"alpha": 1.0, "beta": 1.0}}

    def update(self, model_name, success):
        if model_name not in self.data:
            self.data[model_name] = {"alpha": 1.0, "beta": 1.0}
        if success:
            self.data[model_name]["alpha"] += 1.0
        else:
            self.data[model_name]["beta"] += 1.0
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def weight(self, model_name):
        d = self.data.get(model_name, {"alpha": 1.0, "beta": 1.0})
        return d["alpha"] / (d["alpha"] + d["beta"])