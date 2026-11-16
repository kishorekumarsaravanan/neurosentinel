"""Window-level metrics and the clinical (event-level) alarm logic."""
import numpy as np
import pytest

from neurosentinel.evaluate import window_metrics, alarm_signal, alarm_events, event_metrics


def test_window_metrics_on_known_confusion():
    y = np.array([0] * 90 + [1] * 10)
    p = np.array([0.1] * 85 + [0.9] * 5 + [0.8] * 8 + [0.2] * 2)   # TN 85, FP 5, TP 8, FN 2
    m = window_metrics(y, p)
    assert m["confusion"] == [[85, 5], [2, 8]]
    assert m["accuracy"] == pytest.approx(0.93)
    assert m["sensitivity"] == m["recall"] == pytest.approx(0.8)
    assert m["specificity"] == pytest.approx(85 / 90)
    assert m["precision"] == pytest.approx(8 / 13)
    assert m["f1"] == pytest.approx(2 * (8 / 13) * 0.8 / (8 / 13 + 0.8))
    assert m["balanced_accuracy"] == pytest.approx((0.8 + 85 / 90) / 2)
    assert 0 <= m["roc_auc"] <= 1 and 0 <= m["pr_auc"] <= 1
    assert m["roc"]["fpr"][-1] == 1.0


def test_always_negative_model_has_high_accuracy_but_zero_sensitivity():
    y = np.array([0] * 995 + [1] * 5)
    m = window_metrics(y, np.zeros(1000))
    assert m["accuracy"] > 0.99 and m["sensitivity"] == 0.0 and m["f1"] == 0.0


def test_single_class_labels_skip_auc():
    m = window_metrics(np.zeros(10, int), np.linspace(0, 1, 10))
    assert "roc_auc" not in m and m["sensitivity"] == 0.0


def test_alarm_needs_k_of_n_positive_seconds():
    p = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    # positives in each trailing 4-s window: 1 1 2 2 2 3 3 3 2 1  ->  alarm at indices 5-7
    np.testing.assert_array_equal(alarm_signal(p, thr=0.5, k=3, n=4),
                                  [0, 0, 0, 0, 0, 1, 1, 1, 0, 0])
    assert not alarm_signal(np.full(5, 0.49)).any()        # threshold is inclusive at 0.5
    assert alarm_signal(np.full(5, 0.5)).sum() == 3


def test_alarm_events_merge_within_refractory_period():
    times = np.arange(300, dtype=float)
    alarm = np.zeros(300, bool)
    alarm[[10, 11, 40, 41, 200]] = True                  # 3 rising edges; 10 s and 40 s merge
    assert alarm_events(alarm, times, refractory=60) == [10.0, 200.0]


def test_event_metrics_latency_false_alarms_and_misses():
    times = np.arange(1, 3601, dtype=float)              # one hour at 1 Hz
    p = np.zeros(3600)
    p[(times >= 1006) & (times < 1040)] = 0.9            # detects seizure 1 (onset 1000) 8 s late
    p[(times >= 2500) & (times < 2510)] = 0.9            # false alarm, no seizure there
    rec = {"record": "r1", "times": times, "p": p, "seizures": [(1000.0, 1060.0), (3000.0, 3030.0)], "duration": 3600.0}
    m = event_metrics([rec])
    assert m["seizures"] == 2 and m["detected"] == 1 and m["event_sensitivity"] == 0.5
    assert m["mean_latency"] == pytest.approx(8.0)
    assert m["false_alarms"] == 1 and m["fa_per_hour"] == pytest.approx(1.0)
    assert [s["detected"] for s in m["per_seizure"]] == [True, False]
