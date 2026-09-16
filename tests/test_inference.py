from glclap.inference import frame_times, parse_hotwords


def test_parse_hotwords_deduplicates_in_order():
    assert parse_hotwords(["A", " B "], "A,C\nD") == ["A", "B", "C", "D"]


def test_frame_times_are_monotonic():
    times = frame_times(500)
    assert times
    assert times == sorted(times)
    assert len(times) == 65
