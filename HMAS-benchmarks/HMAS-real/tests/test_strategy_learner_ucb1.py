"""
Tests for core/strategy_learner.py's get_best_hint_ucb1()
(config.UCB1_ENABLED).
"""
from core.strategy_learner import StrategyLearner, ActionStrategy


def make_learner(tmp_path):
    return StrategyLearner(results_dir=str(tmp_path))


def strat(hint, payoff, reward, plan_name, run_number=1):
    return ActionStrategy(
        steps=[hint], payoff=payoff, reward=reward, task_type="cat",
        run_number=run_number, steps_used=3, plan_name=plan_name,
    )


def test_no_records_returns_empty_string(tmp_path):
    learner = make_learner(tmp_path)
    assert learner.get_best_hint_ucb1("cat") == ""


def test_ignores_arms_with_no_success(tmp_path):
    learner = make_learner(tmp_path)
    learner.strategies["cat"] = [
        strat("hint_a", payoff=0.9, reward=0.0, plan_name="planA"),  # high payoff, NEVER succeeded
    ]
    assert learner.get_best_hint_ucb1("cat") == ""


def test_picks_hint_from_only_successful_arm(tmp_path):
    learner = make_learner(tmp_path)
    learner.strategies["cat"] = [
        strat("hint_a", payoff=0.9, reward=0.0, plan_name="planA"),
        strat("hint_b", payoff=0.6, reward=1.0, plan_name="planB"),
    ]
    assert learner.get_best_hint_ucb1("cat") == "hint_b"


def test_excludes_already_tried_hints(tmp_path):
    learner = make_learner(tmp_path)
    learner.strategies["cat"] = [
        strat("hint_a", payoff=0.9, reward=1.0, plan_name="planA"),
        strat("hint_b", payoff=0.5, reward=1.0, plan_name="planB"),
    ]
    assert learner.get_best_hint_ucb1("cat", exclude=["hint_a"]) == "hint_b"


def test_exploration_bonus_favors_under_sampled_arm():
    # Two arms, same total record count (n_total): planA has one
    # high-mean sample, planB has one lower-mean sample but MORE total
    # samples elsewhere is not the case here — construct so the
    # exploration bonus (sqrt(2*ln(N)/n)) actually flips the ranking:
    # planA: 1 sample, mean_payoff=0.60
    # planB: 4 samples, mean_payoff=0.62 (slightly higher mean, but
    #        much smaller exploration bonus since n is larger)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        learner = StrategyLearner(results_dir=d)
        records = [strat("hint_a", payoff=0.60, reward=1.0, plan_name="planA")]
        for _ in range(4):
            records.append(strat("hint_b", payoff=0.62, reward=1.0, plan_name="planB"))
        learner.strategies["cat"] = records

        import math
        n_total = 5
        score_a = 0.60 + math.sqrt(2 * math.log(n_total) / 1)
        score_b = 0.62 + math.sqrt(2 * math.log(n_total) / 4)
        assert score_a > score_b  # sanity check on the scenario itself

        assert learner.get_best_hint_ucb1("cat") == "hint_a"


def test_within_winning_arm_picks_best_reward_then_payoff(tmp_path):
    learner = make_learner(tmp_path)
    learner.strategies["cat"] = [
        strat("hint_a_partial", payoff=0.5, reward=0.5, plan_name="planA", run_number=1),
        strat("hint_a_full",    payoff=0.4, reward=1.0, plan_name="planA", run_number=2),
    ]
    # Both records are the same arm (planA); within the arm, reward wins
    # over payoff (ground truth over heuristic quality).
    assert learner.get_best_hint_ucb1("cat") == "hint_a_full"
