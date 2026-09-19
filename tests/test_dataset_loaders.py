import os
import pytest
from pathlib import Path
from models.data.rsvqa import RSVQADataLoader
from models.data.vrsbench import VRSBenchDataLoader
from models.data.bigearthnet_txt import BigEarthNetTxtDataLoader

def test_rsvqa_loader_sample_fixture():
    fixture_path = "data/datasets/sample_rsvqa.json"
    loader = RSVQADataLoader(annotation_path=fixture_path, images_dir="data/samples", split="lr")
    assert len(loader) == 5
    sample0 = loader[0]
    assert "question" in sample0
    assert "answer" in sample0
    assert "prompt" in sample0
    assert "target" in sample0
    assert sample0["question"] == "Is there any water body present in this area?"
    assert sample0["answer"] == "yes"

def test_rsvqa_loader_missing_path_prints_zenodo_instructions(capsys):
    with pytest.raises(FileNotFoundError, match="RSVQA"):
        RSVQADataLoader(annotation_path="non_existent/path.json")
    captured = capsys.readouterr()
    assert "Zenodo" in captured.err
    assert "zenodo.org/records/6344334" in captured.err

def test_vrsbench_loader_sample_fixture_grounding():
    fixture_path = "data/datasets/sample_vrsbench.json"
    loader = VRSBenchDataLoader(task="grounding", annotation_path=fixture_path, images_dir="data/samples")
    assert len(loader) == 4
    s0 = loader[0]
    assert "expression" in s0
    assert "bbox" in s0
    assert len(s0["bbox"]) == 4
    assert s0["bbox"] == [0.20, 0.15, 0.35, 0.85]
    assert "<|box_start|>" in s0["target"]

def test_vrsbench_loader_missing_path_prints_github_instructions(capsys):
    with pytest.raises(FileNotFoundError, match="VRSBench"):
        VRSBenchDataLoader(annotation_path="non_existent/path.json")
    captured = capsys.readouterr()
    assert "github.com/NJU-SIAT/VRSBench" in captured.err
    assert "CC BY-NC-SA 4.0" in captured.err

def test_bigearthnet_txt_loader_sample_fixture():
    fixture_path = "data/datasets/sample_bigearthnet.txt"
    loader = BigEarthNetTxtDataLoader(annotation_path=fixture_path, images_dir="data/samples")
    assert len(loader) == 3
    s0 = loader[0]
    assert "input" in s0
    assert "output" in s0
    assert "prompt" in s0
    assert "User: <image>\n" in s0["prompt"]
    assert s0["output"] == "Continuous urban fabric, Arable land, Inland waters"

def test_bigearthnet_txt_missing_path_prints_hf_instructions(capsys):
    with pytest.raises(FileNotFoundError, match="BigEarthNet"):
        BigEarthNetTxtDataLoader(annotation_path="non_existent/path.parquet")
    captured = capsys.readouterr()
    assert "BigEarthNet.txt" in captured.err
