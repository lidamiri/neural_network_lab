import pytest

from nnlab.gradcheck import CASES, check_case, run_gradcheck


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_analytic_gradients_match_finite_differences(case):
    result = check_case(case, seed=0)
    assert result["passed"], f"{case['name']} error {result['max_relative_error']:.3e}"


def test_full_gradcheck_suite_passes():
    results = run_gradcheck(seed=1)
    assert len(results) == len(CASES)
    assert all(result["passed"] for result in results)
