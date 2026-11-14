"""Collision audit of the hand + cube scene."""

from manip_rubiks.validation.collisions import check_collisions


def test_collision_audit_passes():
    result = check_collisions(n_seeds=5)
    assert result.passed, result.metrics
    collidable, total = map(int, result.metrics["filters"]["hand_cube_pairs_collidable"].split("/"))
    assert collidable == total > 0
