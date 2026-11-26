import pytest
from src.assistant.providers.offline_provider import OfflineProvider

@pytest.mark.parametrize("prompt,tool,args", [
    ("Which model should I use for the highest benchmark Dice score?", "get_available_models", {}),
    ("Which model is fastest?", "get_available_models", {}),
    ("Change the threshold to 0.7.", "propose_configuration", {"threshold": 0.7}),
    ("Retain my existing settings but remove GrabCut.", "propose_configuration", {"models": ["custom_unet"]}),
    ("Enable uncertainty analysis.", "propose_configuration", {"include_uncertainty": True}),
    ("Compare the Dice, IoU, precision and recall of my current results.", "get_session_context", {}),
    ("Explain the uncertainty around my lesion boundaries.", "analyze_uncertainty", {}),
    ("How would reducing the threshold affect the predicted lesion area?", "analyze_thresholds", {}),
    ("Compare circularity, asymmetry and lesion area.", "get_session_context", {}),
    ("Assess the quality of my uploaded image.", "inspect_image_quality", {}),
    ("Where do the models disagree?", "analyze_disagreement", {}),
])
def test_offline_route(prompt, tool, args):
    response = OfflineProvider().chat([
        {"role": "system", "content": 'Current application state:\n{"configuration":{"models":["custom_unet","grabcut"]}}'},
        {"role": "user", "content": prompt},
    ], tools=[{"function": {"name": tool}}])
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == tool
    assert response.tool_calls[0].arguments == args

@pytest.mark.parametrize("prompt", ["Do not perform segmentation.", "Set threshold to 0.99."])
def test_no_unrequested_change(prompt):
    result = OfflineProvider().chat([{"role":"user","content":prompt}], tools=[{"function":{"name":"propose_configuration"}}])
    assert not result.tool_calls
