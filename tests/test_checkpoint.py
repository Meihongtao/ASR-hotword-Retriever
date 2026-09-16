import torch
from torch import nn
from safetensors.torch import save_file

from glclap.checkpoint import load_adapter_state
from glclap.train_amphion_ddp import checkpoint_payload


def sample_state():
    return {
        "logit_scale": torch.tensor(2.0),
        "audio_adapter.net.0.weight": torch.arange(3, dtype=torch.float32),
        "text_adapter.net.0.weight": torch.arange(4, dtype=torch.float32),
    }


def test_load_safetensors_adapter(tmp_path):
    path = tmp_path / "adapter.safetensors"
    expected = sample_state()
    save_file(expected, str(path))
    actual = load_adapter_state(path)
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], expected[key]) for key in expected)


def test_load_legacy_full_checkpoint_filters_frozen_tower(tmp_path):
    path = tmp_path / "legacy.pt"
    expected = sample_state()
    torch.save({"model": {**expected, "audio.frozen.weight": torch.ones(2)}}, path)
    actual = load_adapter_state(path)
    assert actual.keys() == expected.keys()


class TinyRetriever(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio = nn.Linear(2, 2)
        self.audio_adapter = nn.Linear(2, 2)
        self.text_adapter = nn.Linear(2, 2)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        for parameter in self.audio.parameters():
            parameter.requires_grad = False


def test_training_checkpoint_excludes_frozen_towers():
    model = TinyRetriever()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    payload = checkpoint_payload(model, optimizer, scheduler, {"val_r10": 0.5})
    assert payload["format"] == "glclap-train-state-v2"
    assert set(payload["model"]) == {
        "audio_adapter.weight",
        "audio_adapter.bias",
        "text_adapter.weight",
        "text_adapter.bias",
        "logit_scale",
    }
    assert all(not key.startswith("audio.") for key in payload["model"])
