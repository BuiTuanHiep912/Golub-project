"""Kiểm tra Bảng 2.4 của tài liệu kế hoạch (TV2). Bỏ qua cho đến khi load_golub() được viết."""
import pytest

pytest.importorskip("src.load")


@pytest.mark.skip(reason="Chờ load_golub() — công việc 1.1")
def test_shapes_and_labels():
    from src.load import load_golub
    X, samples = load_golub(layer="cleansed")
    assert X.shape == (72, 7129)
    assert samples["class"].value_counts().to_dict() == {"ALL": 47, "AML": 25}
