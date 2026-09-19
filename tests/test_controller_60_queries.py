import pytest
from backend.scripts.benchmark_controller import run_benchmark, BENCHMARK_CASES

def test_controller_benchmark_suite_has_at_least_60_queries():
    """Requirement 3: Controller test suite: at least 60 labelled queries across all input configurations."""
    assert len(BENCHMARK_CASES) >= 60

def test_controller_benchmark_routing_and_compliance():
    """Requirement 3: Report routing accuracy (>=95%) and parameter-compliance rate (100%) from a real run."""
    summary = run_benchmark(output_json="data/outputs/controller_benchmark_results.json")
    assert summary["total_queries_tested"] >= 60
    assert summary["routing_accuracy_pct"] >= 95.0
    assert summary["parameter_compliance_rate_pct"] == 100.0
    assert summary["incompatible_rejection_accuracy_pct"] == 100.0
