from src.titan.math_core import exact_eval, looks_like_math, plan_and_eval
def test_eval_basic():
    assert exact_eval("37*14")==518
    assert exact_eval("37×14")==518
    assert exact_eval("10+2-3")==9
def test_plan():
    ir = plan_and_eval("12*12")
    assert ir.result==144 and "12*12=144" in ir.steps
def test_detect():
    assert looks_like_math("foo 37×14 bar")
