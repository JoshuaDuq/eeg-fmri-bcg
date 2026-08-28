import numpy as np
import pytest

from bcgnet.compare.qc import (
    alpha_peak_height,
    method_qc_flags,
    remaining_ratio,
)


def test_alpha_peak_height_picks_the_ten_hz_bin() -> None:
    freqs = np.arange(0.0, 40.0, 0.5)
    pxx = np.ones_like(freqs)
    pxx[freqs == 10.0] = 8.0
    pxx[freqs == 12.0] = 3.0
    assert alpha_peak_height(freqs, pxx) == pytest.approx(8.0)


def test_remaining_ratio_flags_added_power() -> None:
    assert remaining_ratio(2.0, 1.0) == pytest.approx(2.0)
    assert remaining_ratio(0.4, 1.0) == pytest.approx(0.4)


def test_qc_flags_prefer_aas_when_bcgnet_adds_power() -> None:
    flags = method_qc_flags(
        remaining_ratios={"delta": 1.6, "theta": 0.5, "alpha": 0.4},
        locked_ratio=0.7,
        alpha_peak_raw=8.0,
        alpha_peak_bcgnet=7.5,
    )
    assert flags["bcgnet_adds_power"] is True
    assert flags["prefer_aas"] is True
    assert flags["bcgnet_locked_worse_than_raw"] is False


def test_qc_flags_do_not_prefer_aas_only_because_alpha_peak_fell() -> None:
    flags = method_qc_flags(
        remaining_ratios={"delta": 0.4, "theta": 0.3, "alpha": 0.3},
        locked_ratio=0.3,
        alpha_peak_raw=20.0,
        alpha_peak_bcgnet=4.0,
    )
    assert flags["alpha_peak_collapsed"] is True
    assert flags["prefer_aas"] is False


def test_qc_flags_prefer_aas_when_locked_residual_increases() -> None:
    flags = method_qc_flags(
        remaining_ratios={"delta": 0.6, "theta": 0.4, "alpha": 0.5},
        locked_ratio=1.2,
        alpha_peak_raw=8.0,
        alpha_peak_bcgnet=7.0,
    )
    assert flags["bcgnet_locked_worse_than_raw"] is True
    assert flags["prefer_aas"] is True


def test_qc_flags_keep_bcgnet_when_harmonics_fall_and_alpha_peak_stays() -> None:
    flags = method_qc_flags(
        remaining_ratios={"delta": 0.6, "theta": 0.3, "alpha": 0.5},
        locked_ratio=0.4,
        alpha_peak_raw=8.0,
        alpha_peak_bcgnet=7.2,
    )
    assert flags["prefer_aas"] is False
    assert flags["alpha_peak_collapsed"] is False
