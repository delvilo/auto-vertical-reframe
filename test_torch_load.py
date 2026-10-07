import ast
from pathlib import Path

def test_torch_load_weights_only():
    auto_reframe_path = Path(__file__).parent / "reframe/saliency/backends/deepgazemr.py"
    with open(auto_reframe_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(auto_reframe_path))

    torch_load_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "load":
                if isinstance(func.value, ast.Name) and func.value.id == "torch":
                    torch_load_calls.append(node)

    assert len(torch_load_calls) == 2, f"Expected 2 torch.load calls, found {len(torch_load_calls)}"

    for call in torch_load_calls:
        weights_only_kw = [kw for kw in call.keywords if kw.arg == "weights_only"]
        assert weights_only_kw, "torch.load call missing weights_only keyword argument"
        assert (
            isinstance(weights_only_kw[0].value, ast.Constant)
            and weights_only_kw[0].value.value is True
        ), "weights_only must be set to True in torch.load call"

