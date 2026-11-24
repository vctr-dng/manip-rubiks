"""Grasp GPU trainer flags: harder training pushes, a critic that sees them, nominal evaluation."""

from manip_rubiks.training import gpu_loop
from manip_rubiks.training.train_grasp_gpu import TASK


def test_flags_default_to_the_nominal_task():
    args = gpu_loop.parse_args(TASK, ["--name", "t"])
    cfg = TASK.make_config(args, 1.0)
    assert cfg.push_scale == 1.0 and not cfg.critic_wrench and cfg.reward.align_barrier == 0.0


def test_training_can_use_harder_pushes_while_evaluation_stays_nominal():
    args = gpu_loop.parse_args(
        TASK, ["--name", "t", "--push-scale", "1.5", "--critic-wrench", "--align-barrier", "2.0"]
    )
    train = TASK.make_config(args, 0.5)
    assert (train.difficulty, train.push_scale, train.critic_wrench, train.reward.align_barrier) == (
        0.5,
        1.5,
        True,
        2.0,
    )
    nominal = TASK.eval_config(TASK.make_config(args, 1.0))
    assert nominal.push_scale == 1.0 and nominal.reward.align_barrier == 0.0
    assert nominal.critic_wrench, "the critic's input must keep the same size in evaluation"
