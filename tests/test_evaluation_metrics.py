import math
import pytest
from models.inference.tiled_inference import compute_box_iou
from models.inference.vlm_wrapper import VLMInferenceWrapper

def test_hand_computed_iou_cases():
    # Case 1: Identical boxes -> IoU = 1.0
    b_ident1 = [0.0, 0.0, 10.0, 10.0]
    b_ident2 = [0.0, 0.0, 10.0, 10.0]
    assert compute_box_iou(b_ident1, b_ident2) == 1.0

    # Case 2: Disjoint boxes -> IoU = 0.0
    b_disj1 = [0.0, 0.0, 5.0, 5.0]
    b_disj2 = [10.0, 10.0, 15.0, 15.0]
    assert compute_box_iou(b_disj1, b_disj2) == 0.0

    # Case 3: Horizontal 50% overlap
    # Box 1: [0, 0, 2, 2], area = 4
    # Box 2: [0, 1, 2, 3], area = 4
    # Intersection: [0, 1, 2, 2], area = 2 * 1 = 2
    # Union: 4 + 4 - 2 = 6
    # IoU: 2 / 6 = 1/3 ~ 0.333333
    b_half1 = [0.0, 0.0, 2.0, 2.0]
    b_half2 = [0.0, 1.0, 2.0, 3.0]
    assert pytest.approx(compute_box_iou(b_half1, b_half2), abs=1e-4) == 1.0 / 3.0

    # Case 4: Completely nested box
    # Outer: [0, 0, 4, 4], area = 16
    # Inner: [1, 1, 3, 3], area = 4
    # Inter = 4, Union = 16
    # IoU = 4 / 16 = 0.25
    b_outer = [0.0, 0.0, 4.0, 4.0]
    b_inner = [1.0, 1.0, 3.0, 3.0]
    assert pytest.approx(compute_box_iou(b_outer, b_inner), abs=1e-4) == 0.25

def test_hand_computed_acc_at_50_and_70():
    ious = [0.85, 0.62, 0.45, 0.15]
    acc_50_cnt = sum(1 for iou in ious if iou >= 0.50)
    acc_70_cnt = sum(1 for iou in ious if iou >= 0.70)

    acc_50 = acc_50_cnt / len(ious)
    acc_70 = acc_70_cnt / len(ious)

    assert acc_50 == 0.50  # 2 out of 4 >= 0.50
    assert acc_70 == 0.25  # 1 out of 4 >= 0.70

def test_hand_computed_confidence_calibration():
    vlm = VLMInferenceWrapper()
    logprobs = [-0.1, -0.2, -0.3]
    avg_logp = sum(logprobs) / len(logprobs)  # -0.2
    expected_conf = round(math.exp(avg_logp), 4)  # exp(-0.2) = 0.81873 -> 0.8187

    computed_conf = vlm.calculate_token_confidence(logprobs)
    assert computed_conf == expected_conf
    assert computed_conf == 0.8187
