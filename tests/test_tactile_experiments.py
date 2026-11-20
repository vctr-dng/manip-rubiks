"""Gate M1 tactile experiments: the rig, the analysis helpers and the gate runner."""

import json

import numpy as np
import pytest

from manip_rubiks.simulation.robot_interface import FINGERTIPS
from manip_rubiks.validation import tactile_m1
from manip_rubiks.validation.tactile_rig import TactileRig


@pytest.fixture(scope="module")
def rig():
    return TactileRig(probe_half=(0.0025, 0.0025, 0.005))


def test_rig_pad_frames_are_orthonormal_and_face_the_probe(rig):
    for f in FINGERTIPS:
        fr = rig.frames[f]
        frame = np.column_stack([fr.normal, fr.along, fr.across])
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-9)
        assert np.linalg.det(frame) > 0
        # The open hand is palm up, so the palmar side faces upwards (the opposed thumb less directly).
        assert fr.normal[2] > (0.5 if f == "thumb" else 0.9)
        assert fr.surface_height(0.0) == 0.0 and fr.surface_height(0.004) < 0.0


def test_pressing_a_probe_lights_only_that_fingertip(rig):
    for f in ("thumb", "index", "pinky"):
        tactile_m1._press(rig, f, 0.0, 0.0, 0.002)
        depth = rig.taxel_si()[..., 0].max(axis=(1, 2))
        assert depth[FINGERTIPS.index(f)] > 1e-4
        assert np.delete(depth, FINGERTIPS.index(f)).max() == 0.0


def test_taxel_depth_grows_with_the_commanded_depth(rig):
    sums = []
    for depth in (0.0005, 0.001, 0.002, 0.003):
        tactile_m1._press(rig, "index", 0.0, 0.0, depth)
        sums.append(rig.taxel_si()[FINGERTIPS.index("index"), ..., 0].sum())
    assert np.all(np.diff(sums) > 0)


def test_slip_channels_separate_along_and_across_motion(rig):
    """A mocap probe would read zero here: the sensor needs a moving body (see the rig docstring)."""
    along = tactile_m1._slide(rig, "index", "along", 0.03)
    across = tactile_m1._slide(rig, "index", "across", 0.03)
    assert along[0] == pytest.approx(0.03, rel=0.05) and along[1] < 0.002
    assert across[1] == pytest.approx(0.03, rel=0.05) and across[0] < 0.002


def test_centroid_recovers_the_press_position(rig):
    k = FINGERTIPS.index("index")
    for along, across in [(-0.003, 0.002), (0.002, -0.003)]:
        tactile_m1._press(rig, "index", along, across, 0.002)
        est = tactile_m1.centroid(rig.taxel_si()[k][..., 0], rig.taxel_positions("index"))
        np.testing.assert_allclose(est, [along, across], atol=0.5e-3)


def test_count_blobs_and_centroid_helpers():
    mask = np.zeros((8, 8), bool)
    assert tactile_m1.count_blobs(mask) == 0
    mask[1, 1:3] = mask[6, 5] = True
    assert tactile_m1.count_blobs(mask) == 2
    mask[2:7, 2] = mask[6, 2:6] = True  # bridges the two
    assert tactile_m1.count_blobs(mask) == 1
    assert np.isnan(tactile_m1.centroid(np.zeros((8, 8)), np.zeros((8, 8, 2)))).all()


def test_quick_gate_passes_and_writes_a_report(tmp_path):
    assert tactile_m1.main(["--quick", "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "m1_report.json").read_text())
    assert report["passed"]
    assert {c["name"] for c in report["checks"]} >= {
        "T1_static_contact",
        "T2_sliding",
        "T3_contact_location",
        "T4_multiple_contacts",
        "T5_robustness",
        "pad_calibration",
    }
    assert "Gate M1 report -- PASSED" in (tmp_path / "m1_report.md").read_text()


@pytest.mark.slow
def test_full_gate_passes():
    report = tactile_m1.run_all(quick=False)
    assert report["passed"], [c["name"] for c in report["checks"] if not c["passed"]]
