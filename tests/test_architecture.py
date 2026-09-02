"""Package boundaries keep method-neutral orchestration out of BCGNet."""

from importlib.util import find_spec


def test_neutral_pipeline_modules_belong_to_bcgstudy() -> None:
    assert find_spec("bcgstudy.discovery") is not None
    assert find_spec("bcgstudy.correction_batch") is not None
    assert find_spec("bcgnet.discovery") is None
    assert find_spec("bcgnet.correction_batch") is None
