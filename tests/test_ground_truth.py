from auto_eval.meta import ground_truth


def test_exact_match_ignores_punct():
    assert ground_truth.exact_match("长江", "长江。") == 1.0
    assert ground_truth.exact_match("长江", "长江！") == 1.0
    assert ground_truth.exact_match("长江", "黄河") == 0.0


def test_f1_range():
    assert ground_truth.f1("长江黄河", "长江") > 0.0
    assert 0.0 <= ground_truth.f1("abc", "xyz") <= 1.0


def test_objective_correct():
    assert ground_truth.compute("长江", "长江")["objective_correct"] == "right"
    assert ground_truth.compute("光合作用释放氧气", "长江")["objective_correct"] == "wrong"
    r = ground_truth.compute("长江的水很长流过很多地方", "长江")
    assert r["objective_correct"] in ("partial", "right", "wrong")
