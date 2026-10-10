"""
Tests for core/strategy_learner.py's per-instance keep_threshold override
(config.LEARNED_PE_WEIGHTS — see core/verifier_learned_weights.py's
calibrate_threshold, wired in agents/rule_based_verifier.py).
"""
from core.strategy_learner import StrategyLearner, ActionStrategy, KEEP_THRESHOLD


def strat(hint, payoff, reward, plan_name, run_number=1):
    return ActionStrategy(
        steps=[hint], payoff=payoff, reward=reward, task_type="cat",
        run_number=run_number, steps_used=3, plan_name=plan_name,
    )


def test_default_keep_threshold_matches_module_constant(tmp_path):
    learner = StrategyLearner(results_dir=str(tmp_path))
    assert learner.keep_threshold == KEEP_THRESHOLD


def test_explicit_none_falls_back_to_module_constant(tmp_path):
    learner = StrategyLearner(results_dir=str(tmp_path), keep_threshold=None)
    assert learner.keep_threshold == KEEP_THRESHOLD


def test_record_run_keep_change_decision_uses_instance_threshold(tmp_path):
    learner = StrategyLearner(results_dir=str(tmp_path), keep_threshold=0.4)
    learner.record_run(task_type="cat", steps=["hint"], payoff=0.5,
                       reward=1.0, steps_used=3, run_number=1)
    # payoff=0.5 >= overridden threshold (0.4) -> stored as usual; the
    # KEEP/CHANGE label itself is only printed, but get_learning_summary's
    # "exploitables" count is a direct, assertable proxy for the same
    # instance-level threshold being used.
    summary = learner.get_learning_summary("cat")
    assert "1/1 exploitables" in summary


def test_learning_summary_exploitables_respects_instance_threshold(tmp_path):
    learner_strict = StrategyLearner(results_dir=str(tmp_path / "a"), keep_threshold=0.9)
    learner_loose  = StrategyLearner(results_dir=str(tmp_path / "b"), keep_threshold=0.1)
    for learner in (learner_strict, learner_loose):
        learner.strategies["cat"] = [strat("hint", payoff=0.5, reward=1.0, plan_name="planA")]
    assert "0/1 exploitables" in learner_strict.get_learning_summary("cat")
    assert "1/1 exploitables" in learner_loose.get_learning_summary("cat")
