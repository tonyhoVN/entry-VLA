import numpy as np

from entry_vla.perception.contact import close_counts, contact_intervals, first_onset, phases_from_contacts


def test_contact_intervals_merge_and_drop():
    f = [0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
    assert contact_intervals(f, min_len=2, merge_gap=1) == [(4, 10)]
    assert contact_intervals(f, min_len=1, merge_gap=0) == [(1, 2), (4, 6), (7, 10), (14, 15)]
    assert contact_intervals([0, 0, 0]) == []


def test_phases_from_contacts():
    assert phases_from_contacts([(5, 9)], 12, "first") == [
        {"phase": "free", "start": 0, "end": 4}, {"phase": "contact_onward", "start": 5, "end": 11}]
    assert [p["phase"] for p in phases_from_contacts([(2, 4), (7, 9)], 12, "all")] == [
        "free", "contact", "free", "contact", "free"]
    assert phases_from_contacts([], 5) == [{"phase": "free", "start": 0, "end": 4}]
    assert first_onset([(3, 6)]) == 3 and first_onset([]) is None


def test_close_counts():
    hand = np.zeros((1, 3))
    env = np.array([[0.01, 0, 0], [0.012, 0, 0], [0.5, 0, 0]])
    assert close_counts(hand, env, 0.015) == (2, 0.01)
    assert close_counts(np.zeros((0, 3)), env, 0.015)[0] == 0
