import ast
from pathlib import Path

def test_torch_load_weights_only():
    auto_reframe_path = Path(__file__).parent / "reframe/saliency/msdb_assets.py"
    with open(auto_reframe_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(auto_reframe_path))

    torch_load_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "load_state_dict_from_url":
                torch_load_calls.append(node)

    assert torch_load_calls, "No explicit checkpoint loading found"

    for call in torch_load_calls:
        weights_only_kw = [kw for kw in call.keywords if kw.arg == "weights_only"]
        assert weights_only_kw, "torch.load call missing weights_only keyword argument"
        assert (
            isinstance(weights_only_kw[0].value, ast.Constant)
            and weights_only_kw[0].value.value is True
        ), "weights_only must be set to True in torch.load call"

