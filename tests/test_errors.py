"""Step checker used by the error analysis (Phase 15)."""
import grpo  # noqa: F401
from grpo.answer_parser import grade
from grpo.errors import classify

P = {"question": "(88 - 58) + 44 = ?", "chain": "88-58=30;30+44=74", "answer": 74, "family": "multi"}


def c(text):
    return classify(text, P, grade(text, P["answer"]))


def test_categories():
    assert c(" 88 - 58 = 30. 30 + 44 = 74.\nFinal answer: <answer>74</answer>") == "faithful"
    assert c(" 44 + 30 = 74.\nFinal answer: <answer>74</answer>") == "lucky"           # step missing
    assert c(" 88 - 58 = 20. 20 + 44 = 74.\nFinal answer: <answer>74</answer>") == "lucky"  # wrong step, right answer
    assert c(" 88 - 58 = 20. 20 + 44 = 64.\nFinal answer: <answer>64</answer>") == "arithmetic"
    assert c(" 88 - 57 = 31. 31 + 44 = 75.\nFinal answer: <answer>75</answer>") == "misread"
    assert c(" 88 + 58 = 146. 146 + 44 = 190.\nFinal answer: <answer>190</answer>") == "wrong_plan"
    assert c(" 88 - 58 = 30. 30 + 44 = 74.\nFinal answer: <answer>75</answer>") == "final_mismatch"
    assert c(" 88 - 58 = 30. 30 + 44 = 74.") == "invalid:no_tag"
    assert c("<answer>12</answer>") == "other"
