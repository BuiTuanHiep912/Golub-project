"""Kiểm thử patient_profile() (TV1, công việc 2.1). Cần tầng curated; chưa có thì bỏ qua."""
import pytest

from src.config import CURATED_DIR

pytestmark = pytest.mark.skipif(not (CURATED_DIR / "X_log10.parquet").exists(),
                                reason="chưa có tầng curated (make curated)")
def test_keys_and_types():
    from src.patient import patient_profile
    p = patient_profile(sample_id=52)
    assert set(p) == {"prediction", "ps", "votes", "similar"}
    assert p["prediction"] in {"ALL", "AML", "uncertain"}
    assert 0 <= p["ps"] <= 1
    assert list(p["votes"].columns) == ["probe_id", "description", "value", "vote", "toward"]
    assert len(p["votes"]) == 50
    assert list(p["similar"].columns) == ["sample_id", "correlation", "class"]
    assert len(p["similar"]) == 5

def test_similar_only_reference():
    from src.load import load_golub
    from src.patient import patient_profile
    _, s = load_golub("curated")
    train = set(s.index[s["split"] == "train"])
    for sid in (52, 1):                        # một mẫu test, một mẫu train
        sim = set(patient_profile(sample_id=sid)["similar"]["sample_id"])
        assert sim <= train and sid not in sim

def test_train_sample_matches_loocv():
    from src.golub_wv import GolubWeightedVoting
    from src.load import load_golub
    from src.patient import patient_profile
    X, s = load_golub("curated")
    ids = s.index[(s["split"] == "train") & (s.index != 1)]
    clf = GolubWeightedVoting(50).fit(X.loc[ids], s.loc[ids, "class"])
    assert patient_profile(sample_id=1)["ps"] == pytest.approx(clf.prediction_strength(X.loc[[1]])[0])

def test_raw_values_same_as_sample_id():
    from src.load import load_golub
    from src.patient import patient_profile
    Xs, _ = load_golub("standardized")         # thang gốc, như đầu ra validate_new_sample()
    a, b = patient_profile(sample_id=52), patient_profile(values=Xs.loc[52])
    assert a["prediction"] == b["prediction"] and a["ps"] == pytest.approx(b["ps"])

def test_bad_arguments():
    from src.patient import patient_profile
    with pytest.raises(ValueError):
        patient_profile()                       # thiếu cả hai
    with pytest.raises(ValueError):
        patient_profile(sample_id=52, values=[1, 2])   # truyền cả hai
    with pytest.raises(KeyError):
        patient_profile(sample_id=999)          # mã không tồn tại

