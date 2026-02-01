# LLM-PathwayCurator/tests/test_shared_gate_mode.py
from llm_pathway_curator import _shared


def test_normalize_gate_mode_synonyms():
    f = _shared.normalize_gate_mode
    assert f("off") == "off"
    assert f("none") == "off"
    assert f("soft") == "note"
    assert f("warn") == "note"
    assert f("hard") == "hard"
    assert f("abstain") == "hard"
    assert f("on") == "hard"
