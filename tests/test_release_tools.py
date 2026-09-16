from pathlib import Path

from tools.release.build_release import source_and_id
from scripts.prepare_data import resolve_public


def test_source_and_id_public_data():
    assert source_and_id("/srv/common-voice-en/en/clips/a.mp3") == (
        "commonvoice",
        "en/clips/a.mp3",
    )
    assert source_and_id("/srv/magicdata_mandarin_read/train/s/u.wav") == (
        "magicdata",
        "train/s/u.wav",
    )


def test_source_and_id_release_audio():
    assert source_and_id(
        "/srv/stage2_hotword_datasets/ContextASR-Bench/audio/x.wav"
    ) == ("contextasr", "ContextASR-Bench/audio/x.wav")
    assert source_and_id(
        "/srv/stage2_hotword_datasets/domain_TTS/TTS/math/m1/meeting.wav"
    ) == ("domain_tts_full_meeting", "domain_TTS/TTS/math/m1/meeting.wav")
    assert source_and_id("/srv/datas/极客湾_dataset/wavs/x.wav") == (
        "jikewan_video",
        "极客湾_dataset/wavs/x.wav",
    )


def test_prompt_hash_matches_release_prompt():
    import hashlib

    prompt = Path(__file__).parents[1] / "tools/entity_extract/hotword_prompt.txt"
    assert hashlib.sha256(prompt.read_bytes()).hexdigest() == (
        "ababcea09f10ab1a7e2ad2443ae17de359cd74eaa2208043ce17f14089286b23"
    )


def test_resolve_public_accepts_corpus_or_split_root(tmp_path):
    cv_clip = tmp_path / "cv" / "clips" / "a.mp3"
    cv_clip.parent.mkdir(parents=True)
    cv_clip.touch()
    assert resolve_public(tmp_path / "cv", "en/clips/a.mp3", "commonvoice") == cv_clip

    md_wav = tmp_path / "magic" / "train" / "spk" / "a.wav"
    md_wav.parent.mkdir(parents=True)
    md_wav.touch()
    assert resolve_public(tmp_path / "magic", "train/spk/a.wav", "magicdata") == md_wav
