import numpy as np
import soundfile as sf

from glclap.data import EntityBatchSampler, _load_audio_16k


class Rows:
    rows = [
        {"entities": ["a"]},
        {"entities": ["a", "b"]},
        {"entities": ["c"]},
        {"entities": ["d"]},
    ]


def test_entity_batches_have_no_within_batch_conflict():
    sampler = EntityBatchSampler(Rows(), batch_size=3, seed=7)
    for batch in sampler:
        entities = []
        for index in batch:
            entities.extend(Rows.rows[index]["entities"])
        assert len(entities) == len(set(entities))


def test_stereo_audio_is_mixed_to_mono(tmp_path):
    path = tmp_path / "stereo.wav"
    audio = np.stack(
        [np.ones(320, dtype=np.float32), np.zeros(320, dtype=np.float32)], axis=1
    )
    sf.write(path, audio, 16000, subtype="FLOAT")
    result = _load_audio_16k(path)
    assert result.shape == (320,)
    assert np.allclose(result.numpy(), 0.5)
