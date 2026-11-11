"""
Require causal evidence before converting pytest startup errors to model outcomes.
"""

import hashlib
import json
import sys
from difflib import unified_diff
from pathlib import Path

import pytest
from examples.mini_swe.benchmark_grading import (
    TEST_OUTPUT_END,
    TEST_OUTPUT_START,
    model_collection_failure_candidate,
    parse_benchmark_report,
)
from examples.mini_swe.syntax_probe import SYNTAX_REGRESSION_MARKER

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def task():
    return {
        "instance_id": "example__repo-1",
        "repo": "example/repo",
        "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_keep.py::test_keep"],
    }


@pytest.fixture
def baseline():
    return {
        "resolved_by": "harness",
        "resolved": False,
        "f2p_pass": 0,
        "p2p_pass": 1,
        "test_results_observed": 2,
        "baseline_image_id": "sha256:" + "a" * 64,
    }


def patch(before: str = "value = 1\n", after: str = "value = missing\n", path: str = "package/model.py") -> str:
    """
    Construct source and target coordinates from actual source text.
    """
    return f"diff --git a/{path} b/{path}\n" + "".join(
        unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), f"a/{path}", f"b/{path}")
    )


def output(error_type="NameError", path="/testbed/package/model.py", prefix="", failure="", startup=True):
    context = "ImportError while loading conftest '/testbed/tests/conftest.py'.\n" if startup else ""
    return (
        f"{TEST_OUTPUT_START}\n{prefix}{context}{failure}"
        f"{path}:1: in <module>\n    value = missing\nE   {error_type}: Model startup failed\n"
        f"{TEST_OUTPUT_END}\n"
    )


@pytest.mark.parametrize(
    "error_type",
    [
        "NameError",
        "ValueError",
        "AttributeError",
        "TypeError",
        "ImportError",
        "ModuleNotFoundError",
        "RuntimeError",
        "SyntaxError",
        "Exception",
        "CustomFailure",
    ],
)
def test_startup_error_needs_a_task_baseline_or_compile_proof(task, baseline, error_type):
    text = output(error_type)
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch())
    result = parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)

    assert not result["resolved"], "A proven startup regression cannot resolve the task."
    assert result["f2p_pass"] == result["p2p_pass"] == 0, (
        "Official aggregation must count the missing task tests as failures."
    )
    assert result["model_failure"]["error_type"] == error_type, (
        "Attribution must not depend on an exception allowlist."
    )
    assert result["model_failure"]["attribution"] == "baseline_report", "Record the source of causal evidence."
    assert result["model_failure"]["baseline_test_results_observed"] == 2, (
        "A baseline must actually execute task tests."
    )


@pytest.mark.parametrize(
    "change",
    [
        {"test_results_observed": 0},
        {"resolved_by": "apply_failed"},
        {"model_failure": {"error_type": "NameError"}},
        {"baseline_image_id": ""},
    ],
)
def test_incomplete_baseline_does_not_prove_a_model_failure(task, baseline, change):
    baseline.update(change)
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, output(), "gym", model_patch=patch(), baseline_report=baseline)


@pytest.mark.parametrize(
    "path",
    [
        "/opt/conda/lib/python3.10/site-packages/package/model.py",
        "/testbed/.venv/site-packages/package/model.py",
        "/testbed/../package/model.py",
        "/different/package/model.py",
    ],
)
def test_baseline_does_not_attribute_an_external_terminal_frame(task, baseline, path):
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, output(path=path), "gym", model_patch=patch(), baseline_report=baseline)


@pytest.mark.parametrize(
    "model_patch", ["", "not a patch", "--- a/package/model.py\n+++ b/package/model.py\n@@ broken"]
)
def test_missing_or_malformed_patch_cannot_explain_a_startup_failure(task, baseline, model_patch):
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, output(), "gym", model_patch=model_patch, baseline_report=baseline)


@pytest.mark.parametrize(
    "failure",
    [
        "ERROR: No matching distribution found for dependency==0.1",
        "ERROR: Could not find a version that satisfies the requirement dependency",
        "ERROR: Failed building wheel for dependency",
        "ERROR: Could not build wheels for dependency",
        "ERROR: Could not install packages due to an OSError",
        "error: subprocess-exited-with-error",
        "TransportError: Failed to execute Docker command",
        "ConnectionError: Connection refused",
        "TimeoutError: Evaluation timed out",
        "subprocess.TimeoutExpired: Command pytest exceeded its deadline",
        ">>>>> Tests Timed Out",
    ],
)
def test_test_phase_infrastructure_failure_remains_an_error(task, baseline, failure):
    text = output(failure=failure + "\n")
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)


@pytest.mark.parametrize("marker", [">>>>> Tests Timed Out", ">>>>> Tests Errored", ">>>>> Reset Failed"])
def test_official_execution_failure_markers_cannot_be_overridden(task, baseline, marker):
    text = output(prefix=marker + "\n")
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)


def test_nonfatal_setup_messages_require_later_causal_evidence(task, baseline):
    text = output(prefix="ERROR: No matching distribution found for an unused extra\n+ pytest tests\n")
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch())
    result = parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)

    assert not result["resolved"], "A task baseline distinguishes incidental setup output from model source failure."


def test_external_terminal_frame_is_not_replaced_with_a_modified_caller(task, baseline):
    text = output(path="/opt/site-packages/dependency.py", prefix="package/model.py:1: in repair\n")
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)


def test_unknown_output_and_missing_markers_never_gain_model_rewards(task, baseline):
    for text in ["", "all tests passed", output().replace(TEST_OUTPUT_END, ""), output(startup=False)]:
        with pytest.raises(ValueError, match="markers|No parseable"):
            parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)


def test_normal_reports_and_collection_reports_keep_official_semantics(task):
    text = (
        f"{TEST_OUTPUT_START}\nPASSED tests/test_fix.py::test_fix\n"
        f"PASSED tests/test_keep.py::test_keep\n{TEST_OUTPUT_END}\n"
    )
    result = parse_benchmark_report(task, text, "gym")
    assert result["resolved"], "Valid reports retain official resolution rules."
    assert result["test_results_observed"] == 2, "Count observed task results separately from expected totals."

    text = f"{TEST_OUTPUT_START}\nERROR collecting tests/test_fix.py\n{TEST_OUTPUT_END}\n"
    result = parse_benchmark_report(task, text, "gym")
    assert not result["resolved"], "Complete collection reports retain official unresolved semantics."
    assert result["test_results_observed"] == 0, "An empty collection report cannot prove baseline readiness."


def test_candidate_identification_does_not_assign_an_outcome():
    candidate = model_collection_failure_candidate(output(), patch())
    assert candidate["path"] == "package/model.py", "Only task source candidates require a baseline control."
    assert "attribution" not in candidate, "A candidate is not evidence that the task environment works."


def test_removed_class_method_requires_a_complete_counterfactual_task_report(task, baseline):
    model_patch = patch(
        "class Categorical:\n    def reorder_categories(self):\n        return self\n",
        "class Categorical:\n    pass\n",
        "pandas/core/arrays/categorical.py",
    )
    text = (
        f"{TEST_OUTPUT_START}\n"
        "ImportError while loading conftest '/testbed/pandas/conftest.py'.\n"
        "pandas/core/indexes/category.py:80: in <module>\n"
        "    class CategoricalIndex(NDArrayBackedExtensionIndex):\n"
        "pandas/core/indexes/extension.py:127: in wrapper\n"
        "    meth = _inherit_from_data(name, delegate, cache=cache, wrap=wrap)\n"
        "pandas/core/indexes/extension.py:52: in _inherit_from_data\n"
        "    attr = getattr(delegate, name)\n"
        "E   AttributeError: type object 'Categorical' has no attribute 'reorder_categories'\n"
        f"{TEST_OUTPUT_END}\n"
    )
    candidate = model_collection_failure_candidate(text, model_patch)

    assert candidate is not None, "A class definition can affect import code without appearing in its traceback."
    assert candidate["path"] == "pandas/core/indexes/extension.py", "Record the actual failure location."
    assert candidate["line"] == 52, "Do not attach the traceback line to the modified class source."
    assert candidate["modified_sources"] == ["pandas/core/arrays/categorical.py"], (
        "Keep model source changes separate from the observed failure location."
    )
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=model_patch)
    result = parse_benchmark_report(task, text, "gym", model_patch=model_patch, baseline_report=baseline)

    assert not result["resolved"], "A proven model startup regression remains an official failed task."
    assert result["model_failure"]["attribution"] == "baseline_report", "Require task execution in the control."


def test_a_partial_positive_baseline_cannot_prove_task_readiness(task, baseline):
    baseline["test_results_observed"] = 1
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, output(), "gym", model_patch=patch(), baseline_report=baseline)


def downstream_startup_output(error_type="ValueError", terminal="/testbed/package/downstream.py"):
    """
    Reproduce a changed caller failing in an unchanged repository implementation.
    """
    return (
        f"{TEST_OUTPUT_START}\n"
        "ImportError while loading conftest '/testbed/tests/conftest.py'.\n"
        "package/model.py:12: in construct\n"
        "    return downstream(value)\n"
        f"{terminal}:27: in downstream\n"
        "    raise ValueError('Unsupported value')\n"
        f"E   {error_type}: Unsupported value\n"
        f"{TEST_OUTPUT_END}\n"
    )


@pytest.mark.parametrize("error_type", ["ValueError", "TypeError", "AttributeError", "NameError", "RuntimeError"])
def test_startup_control_keeps_terminal_location_and_changed_sources(task, baseline, error_type):
    text = downstream_startup_output(error_type)
    candidate = model_collection_failure_candidate(text, patch())

    assert candidate is not None, "A changed caller may identify a control candidate within pytest startup."
    assert candidate["path"] == "package/downstream.py", "Record where the exception actually occurred."
    assert candidate["line"] == 27, "Keep the failure coordinate attached to its actual source."
    assert candidate["modified_sources"] == ["package/model.py"], "List source changes without inventing a location."
    assert "attribution" not in candidate, "A changed frame alone cannot prove a model outcome."
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch())
    result = parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)

    assert not result["resolved"], "A proven startup regression cannot resolve a task."
    assert result["model_failure"]["attribution"] == "baseline_report", "Require a successful task control."
    assert result["model_failure"]["error_type"] == error_type, "Preserve the actual downstream exception."


def test_startup_candidate_keeps_all_source_changes_separate_from_terminal_location():
    model_patch = patch() + patch(path="package/wrapper.py")
    text = downstream_startup_output().replace(
        "package/model.py:12: in construct\n", "package/wrapper.py:3: in wrap\npackage/model.py:12: in construct\n"
    )
    candidate = model_collection_failure_candidate(text, model_patch)

    assert candidate["path"] == "package/downstream.py", "Keep the observed failure site unchanged."
    assert candidate["line"] == 27, "Do not borrow coordinates from a changed caller."
    assert candidate["modified_sources"] == ["package/model.py", "package/wrapper.py"], (
        "The control applies to the complete source patch."
    )


@pytest.mark.parametrize(
    "terminal",
    ["/opt/site-packages/dependency.py", "/testbed/.venv/site-packages/dependency.py", "/testbed/../outside.py"],
)
def test_modified_startup_caller_cannot_explain_external_terminal_errors(task, baseline, terminal):
    text = downstream_startup_output(terminal=terminal)
    assert model_collection_failure_candidate(text, patch()) is None, "External failure sites remain unproven."
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)


@pytest.mark.parametrize("change", [{"test_results_observed": 0}, {"baseline_image_id": ""}])
def test_modified_startup_caller_cannot_use_an_incomplete_baseline(task, baseline, change):
    baseline.update(change)
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, downstream_startup_output(), "gym", model_patch=patch(), baseline_report=baseline)


def test_modified_startup_caller_does_not_override_transport_failure(task, baseline):
    text = downstream_startup_output("ConnectionError")
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)


def full_startup_trace(error, path="/testbed/package/model.py", line=1, loader="_importconftest"):
    """
    Format pytest's configuration loading traceback without its short conftest diagnostic.
    """
    return (
        "Traceback (most recent call last):\n"
        '  File "/opt/conda/lib/python3.10/site-packages/_pytest/config/__init__.py", '
        f"line 659, in {loader}\n"
        "    mod = import_path(conftestpath)\n"
        '  File "/testbed/tests/conftest.py", line 23, in <module>\n'
        "    import package.model\n"
        f'  File "{path}", line {line}, in <module>\n'
        f"{error}\n"
    )


@pytest.mark.parametrize("loader", ["_importconftest", "_loadconftestmodules"])
def test_full_pytest_trace_needs_verified_baseline_and_preserves_setup_phase_boundaries(task, baseline, loader):
    prefix = (
        "Traceback (most recent call last):\n"
        '  File "/opt/installer.py", line 1, in install\n'
        "ConnectionError: Could not fetch an unused dependency\n"
        "ERROR: No matching distribution found for an unused extra\n"
        "+ pytest tests\n"
    )
    trace = full_startup_trace("NameError: Model startup failed", loader=loader)
    text = f"{TEST_OUTPUT_START}\n{prefix}{trace}{TEST_OUTPUT_END}\n"
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch())
    result = parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)

    assert result["model_failure"]["attribution"] == "baseline_report", "Require observed baseline test evidence."
    assert result["model_failure"]["error_type"] == "NameError", "Attribute the actual test startup exception."
    assert result["test_results_observed"] == 0, "Startup failures must not manufacture observed test results."


def test_full_conftest_import_failure_trace_identifies_pytest_configuration_phase(task, baseline):
    trace = full_startup_trace("NameError: Model startup failed", loader="load_plugin")
    trace += "_pytest.config.ConftestImportFailure: NameError from tests/conftest.py\n"
    text = f"{TEST_OUTPUT_START}\n{trace}{TEST_OUTPUT_END}\n"
    result = parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)

    assert result["model_failure"]["error_type"] == "NameError", "Keep the source exception within its full traceback."


def test_bokeh_full_configuration_trace_accepts_matching_compile_proof(task):
    path = "src/bokeh/models/renderers/glyph_renderer.py"
    before = "\n" * 236 + (
        "class GlyphRenderer:\n"
        "    def construct_color_bar(self):\n"
        "        if True:\n"
        "            return ColorBar(color_mapper=fill_color.transform, **kwargs)\n"
    )
    after = before.replace("            return ColorBar", "        return ColorBar")
    model_patch = patch(before, after, path)
    compile(before, path, "exec", dont_inherit=True)
    with pytest.raises(IndentationError) as failure:
        compile(after, path, "exec", dont_inherit=True)
    error = failure.value
    evidence = {
        "path": path,
        "error_type": type(error).__name__,
        "line": error.lineno,
        "message": error.msg,
        "offset": error.offset,
        "revision": "d5ce3ec088f06bfaeac1e553bb42b4b88c069f58",
        "model_patch_sha256": hashlib.sha256(model_patch.encode()).hexdigest(),
        "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }
    trace = full_startup_trace(
        f"IndentationError: {error.msg}", path=f"/testbed/{path}", line=240, loader="_loadconftestmodules"
    )
    text = f"{TEST_OUTPUT_START}\n{trace}{SYNTAX_REGRESSION_MARKER}{json.dumps(evidence)}\n{TEST_OUTPUT_END}\n"
    result = parse_benchmark_report(task, text, "gym", model_patch=model_patch)

    assert not result["resolved"], "A proven model indentation failure remains unresolved."
    assert result["model_failure"]["attribution"] == "syntax_regression", "Use exact source compile evidence."
    assert result["model_failure"]["line"] == 240, "Preserve the observed bokeh traceback line."


@pytest.mark.parametrize("loader", ["load_plugin", "pytest_runtest_call"])
def test_non_conftest_tracebacks_cannot_gain_model_rewards(task, baseline, loader):
    trace = full_startup_trace("NameError: Model startup failed", loader=loader)
    text = f"{TEST_OUTPUT_START}\n{trace}{TEST_OUTPUT_END}\n"
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)


def test_arbitrary_application_traceback_cannot_claim_a_pytest_loading_phase(task, baseline):
    trace = full_startup_trace("NameError: Model startup failed").replace(
        "/opt/conda/lib/python3.10/site-packages/_pytest/config/__init__.py", "/testbed/package/loader.py"
    )
    text = f"{TEST_OUTPUT_START}\n{trace}{TEST_OUTPUT_END}\n"
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)


def test_full_pytest_trace_keeps_test_phase_transport_errors_fail_closed(task, baseline):
    trace = full_startup_trace("NameError: Model startup failed").replace(
        "    mod = import_path(conftestpath)\n", "    mod = import_path(conftestpath)\nTransportError: Docker failed\n"
    )
    text = f"{TEST_OUTPUT_START}\n{trace}{TEST_OUTPUT_END}\n"
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)


def test_official_execution_failure_marker_outside_test_frame_is_still_fatal(task, baseline):
    trace = full_startup_trace("NameError: Model startup failed")
    text = f">>>>> Tests Timed Out\n{TEST_OUTPUT_START}\n{trace}{TEST_OUTPUT_END}\n"
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)


@pytest.mark.parametrize(
    "marker", [">>>>> Tests Timed Out", ">>>>> Tests Errored", ">>>>> Patch Apply Failed", ">>>>> Reset Failed"]
)
def test_observed_tests_cannot_override_official_execution_failure_markers(task, marker):
    text = (
        f"{marker}\n{TEST_OUTPUT_START}\nPASSED tests/test_fix.py::test_fix\n"
        f"PASSED tests/test_keep.py::test_keep\n{TEST_OUTPUT_END}\n"
    )
    with pytest.raises(ValueError, match="Evaluation execution failed"):
        parse_benchmark_report(task, text, "gym")


@pytest.fixture
def cython_output():
    from pathlib import Path

    tail = (Path(__file__).parent / "fixtures/pandas_cython_startup_failure.txt").read_text()
    return f"{TEST_OUTPUT_START}\n{tail}"


def cython_patch():
    return patch("cdef int value = 1\n", "cdef int value =\n", "pandas/_libs/lib.pyx")


def test_real_cython_startup_tail_requires_a_successful_task_baseline(task, baseline, cython_output):
    candidate = model_collection_failure_candidate(cython_output, cython_patch())

    assert candidate is not None, "An explicit compiler diagnostic may identify a baseline control candidate."
    assert candidate["path"] == "pandas/_libs/lib.pyx", "The compiler must name the modified Cython source."
    assert candidate["line"] == 746, "Retain the actual compiler diagnostic coordinate."
    assert "attribution" not in candidate, "A compiler diagnostic cannot establish baseline readiness."
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, cython_output, "gym", model_patch=cython_patch())
    result = parse_benchmark_report(task, cython_output, "gym", model_patch=cython_patch(), baseline_report=baseline)

    assert not result["resolved"], "A proven model build regression must remain unresolved."
    assert result["f2p_pass"] == result["p2p_pass"] == 0, "Official missing task statuses remain failures."
    assert result["model_failure"]["error_type"] == "CythonCompileError", "Retain the compiler failure category."
    assert result["model_failure"]["attribution"] == "baseline_report", "Cython failures always require a baseline."
    assert result["model_failure"]["baseline_image_id"] == baseline["baseline_image_id"], (
        "The control must identify the immutable evaluated image."
    )


@pytest.mark.parametrize(
    "before,after",
    [
        ("Error compiling Cython file:", "Build progress:"),
        ("ninja: build stopped: subcommand failed.", "Build complete."),
        ("/opt/miniconda3/envs/testbed/bin/ninja", "/opt/miniconda3/envs/testbed/bin/make"),
        ("ImportError while loading conftest", "ImportError outside pytest startup"),
        ("/testbed/pandas/_libs/lib.pyx:746:0:", "/testbed/pandas/_libs/other.pyx:746:0:"),
        ("/testbed/pandas/_libs/lib.pyx:746:0:", "/opt/dependency/lib.pyx:746:0:"),
        ("/testbed/pandas/_libs/lib.pyx:746:0:", "/testbed/../lib.pyx:746:0:"),
        ("/testbed/pandas/_libs/lib.pyx:746:0:", "/testbed/pandas/_libs/lib.pyx:0:0:"),
        (TEST_OUTPUT_END, ""),
    ],
)
def test_cython_attribution_requires_the_complete_modified_source_build_chain(
    task, baseline, cython_output, before, after
):
    with pytest.raises(ValueError, match="markers|No parseable"):
        parse_benchmark_report(
            task, cython_output.replace(before, after), "gym", model_patch=cython_patch(), baseline_report=baseline
        )


@pytest.mark.parametrize(
    "change",
    [
        {"test_results_observed": 0},
        {"resolved_by": "apply_failed"},
        {"model_failure": {"error_type": "CythonCompileError"}},
        {"baseline_image_id": ""},
    ],
)
def test_cython_build_failures_cannot_use_an_incomplete_baseline(task, baseline, cython_output, change):
    baseline.update(change)
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, cython_output, "gym", model_patch=cython_patch(), baseline_report=baseline)


@pytest.mark.parametrize(
    "failure",
    [
        "ERROR: Could not build wheels for dependency",
        "ConnectionError: Connection refused",
        "subprocess.TimeoutExpired: Build timed out",
        ">>>>> Tests Timed Out",
        ">>>>> Tests Errored",
        ">>>>> Patch Apply Failed",
        ">>>>> Reset Failed",
    ],
)
def test_cython_diagnostics_do_not_override_infrastructure_failures(task, baseline, cython_output, failure):
    text = cython_output.replace(
        "ImportError while loading conftest", failure + "\nImportError while loading conftest"
    )
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=cython_patch(), baseline_report=baseline)


def test_a_cython_diagnostic_cannot_explain_a_python_only_patch(task, baseline, cython_output):
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, cython_output, "gym", model_patch=patch(), baseline_report=baseline)


@pytest.fixture
def import_failure_output():
    from pathlib import Path

    tail = (Path(__file__).parent / "fixtures/dvc_import_startup_failure.txt").read_text()
    return f"{TEST_OUTPUT_START}\n{tail}"


def import_failure_patch():
    return patch("def ask_password():\n    return ''\n", "value = 1\n", "dvc/tree/http.py")


def test_explicit_imported_source_requires_a_task_baseline(task, baseline, import_failure_output):
    candidate = model_collection_failure_candidate(import_failure_output, import_failure_patch())

    assert candidate is not None, "An exact imported module path may identify a baseline control candidate."
    assert candidate["path"] == "dvc/tree/webdav.py", "Keep the actual importing source from the traceback."
    assert candidate["line"] == 16, "Keep the traceback coordinate attached to its actual source."
    assert candidate["modified_sources"] == ["dvc/tree/http.py"], "Record changed sources separately."
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, import_failure_output, "gym", model_patch=import_failure_patch())
    result = parse_benchmark_report(
        task, import_failure_output, "gym", model_patch=import_failure_patch(), baseline_report=baseline
    )

    assert not result["resolved"], "A proven model import regression must remain unresolved."
    assert result["model_failure"]["attribution"] == "baseline_report", (
        "An import message alone cannot assign a reward."
    )


@pytest.mark.parametrize(
    "before,after",
    [
        ("ImportError while loading conftest", "ImportError outside pytest startup"),
        (TEST_OUTPUT_END, ""),
    ],
)
def test_import_diagnostic_requires_a_completed_pytest_startup_report(
    task, baseline, import_failure_output, before, after
):
    text = import_failure_output.replace(before, after)
    if "outside pytest startup" in text:
        text = text.replace("ConftestImportFailure:", "UnrelatedFailure:")
    with pytest.raises(ValueError, match="markers|No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=import_failure_patch(), baseline_report=baseline)


@pytest.mark.parametrize(
    "before,after",
    [
        ("(/testbed/dvc/tree/http.py)", "(unknown location)"),
        ("(/testbed/dvc/tree/http.py)", "(/opt/site-packages/dvc/tree/http.py)"),
        ("(/testbed/dvc/tree/http.py)", "(/testbed/dvc/tree/other.py)"),
        ("E   ImportError: cannot import name", "E   RuntimeError: cannot import name"),
        ("E   ImportError: cannot import name", "E   ImportError: another failure involving name"),
    ],
)
def test_startup_control_does_not_depend_on_exception_symbols(task, baseline, import_failure_output, before, after):
    text = import_failure_output.replace(before, after)
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=import_failure_patch())
    result = parse_benchmark_report(task, text, "gym", model_patch=import_failure_patch(), baseline_report=baseline)

    assert not result["resolved"], "A complete counterfactual control can prove an import startup regression."
    assert result["model_failure"]["path"] == "dvc/tree/webdav.py", "Keep the observed repository failure location."
    assert result["model_failure"]["attribution"] == "baseline_report", "Exception text cannot substitute for proof."


def test_unmodified_repository_failure_location_still_requires_a_complete_control(task, baseline):
    text = output(path="/testbed/package/unmodified.py")
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch())
    result = parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)

    assert result["model_failure"]["path"] == "package/unmodified.py", "Record the actual exception source."
    assert result["model_failure"]["modified_sources"] == ["package/model.py"], (
        "A modified definition need not appear in its caller's traceback."
    )


def test_a_python_startup_control_requires_an_actual_python_source_change(task, baseline):
    model_patch = patch(path="README.txt")
    assert model_collection_failure_candidate(output(), model_patch) is None, (
        "Non Python edits do not explain imports."
    )
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, output(), "gym", model_patch=model_patch, baseline_report=baseline)


@pytest.mark.parametrize("failure", ["ConnectionError: Connection refused", "subprocess.TimeoutExpired: timed out"])
def test_import_diagnostic_does_not_override_startup_infrastructure_errors(
    task, baseline, import_failure_output, failure
):
    text = import_failure_output.replace("tests/conftest.py:6:", failure + "\ntests/conftest.py:6:")
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=import_failure_patch(), baseline_report=baseline)


def test_import_diagnostic_requires_observed_baseline_tests(task, baseline, import_failure_output):
    baseline["test_results_observed"] = 0
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(
            task, import_failure_output, "gym", model_patch=import_failure_patch(), baseline_report=baseline
        )


@pytest.fixture
def zero_collection_output():
    return (
        f"{TEST_OUTPUT_START}\n"
        "============================= test session starts ==============================\n"
        "platform linux -- Python 3.12.4, pytest-8.3.2, pluggy-1.5.0\n"
        "rootdir: /testbed\n"
        "configfile: setup.cfg\n"
        "plugins: cov-5.0.0, xdist-3.6.1\n"
        "collected 0 items\n\n"
        "============================ no tests ran in 0.46s =============================\n"
        f"{TEST_OUTPUT_END}\n"
    )


def test_zero_collection_requires_a_complete_task_baseline(task, baseline, zero_collection_output):
    model_patch = patch()
    candidate = model_collection_failure_candidate(zero_collection_output, model_patch)
    assert candidate is not None, "Completed empty pytest collection must request a task baseline."
    assert candidate["error_type"] == "NoTestsCollected", "Record pytest's collection outcome."
    assert candidate["path"] is candidate["line"] is None, "Empty collection supplies no exception source frame."
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, zero_collection_output, "gym", model_patch=model_patch)
    result = parse_benchmark_report(
        task, zero_collection_output, "gym", model_patch=model_patch, baseline_report=baseline
    )
    assert not result["resolved"], "Removing test collection cannot resolve a task."
    assert result["f2p_pass"] == result["p2p_pass"] == result["test_results_observed"] == 0, (
        "Official aggregation counts missing tests as failures."
    )
    assert result["model_failure"]["attribution"] == "baseline_report", "Require independent readiness evidence."


@pytest.mark.parametrize(
    "change",
    [
        {"test_results_observed": 0},
        {"test_results_observed": 1},
        {"test_results_observed": True},
        {"resolved_by": "apply_failed"},
        {"baseline_image_id": ""},
        {"model_failure": {"error_type": "NoTestsCollected"}},
    ],
)
def test_zero_collection_cannot_use_an_incomplete_baseline(task, baseline, zero_collection_output, change):
    baseline.update(change)
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, zero_collection_output, "gym", model_patch=patch(), baseline_report=baseline)


@pytest.mark.parametrize(
    "removed",
    [
        TEST_OUTPUT_START,
        TEST_OUTPUT_END,
        "============================= test session starts ==============================",
        "collected 0 items",
        "============================ no tests ran in 0.46s =============================",
    ],
)
def test_zero_collection_needs_a_completed_pytest_session(task, baseline, zero_collection_output, removed):
    text = zero_collection_output.replace(removed, "")
    assert model_collection_failure_candidate(text, patch()) is None, (
        "Partial output cannot prove collection finished."
    )
    with pytest.raises(ValueError, match="markers|No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)


@pytest.mark.parametrize("model_patch", ["", "not a patch", patch(path="README.txt")])
def test_zero_collection_requires_modified_python(task, baseline, zero_collection_output, model_patch):
    assert model_collection_failure_candidate(zero_collection_output, model_patch) is None, (
        "A source change is required before requesting a counterfactual control."
    )
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, zero_collection_output, "gym", model_patch=model_patch, baseline_report=baseline)


@pytest.mark.parametrize(
    "failure",
    [
        "ConnectionError: Connection refused",
        "subprocess.TimeoutExpired: timed out",
        ">>>>> Tests Errored",
        ">>>>> Tests Timed Out",
        ">>>>> Patch Apply Failed",
        ">>>>> Reset Failed",
    ],
)
def test_zero_collection_cannot_override_execution_failures(task, baseline, zero_collection_output, failure):
    text = zero_collection_output.replace("collected 0 items", failure + "\ncollected 0 items")
    assert model_collection_failure_candidate(text, patch()) is None, "Execution errors remain grading failures."
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)


def test_zero_collection_setup_noise_still_requires_a_complete_baseline(task, baseline, zero_collection_output):
    text = zero_collection_output.replace(
        TEST_OUTPUT_START, TEST_OUTPUT_START + "\nERROR: No matching distribution found for an unused extra"
    )
    result = parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)
    assert result["model_failure"]["baseline_test_results_observed"] == 2, (
        "Only independent task results can establish readiness after setup noise."
    )


@pytest.mark.parametrize("error_type", ["ModuleNotFoundError", "RuntimeError", "Exception"])
def test_short_conftest_diagnostic_requires_a_complete_task_baseline(task, baseline, error_type):
    text = (
        f"{TEST_OUTPUT_START}\n"
        "ImportError while loading conftest '/testbed/pandas/conftest.py'.\n"
        f"{error_type}: No module named 'pandas.conftest'\n"
        f"{TEST_OUTPUT_END}\n"
    )
    candidate = model_collection_failure_candidate(text, patch())
    assert candidate is not None, "Pytest's short conftest diagnostic must request a task baseline."
    assert candidate["path"] == "pandas/conftest.py" and candidate["line"] is None, (
        "Record the named conftest without inventing an exception line."
    )
    assert candidate["error_type"] == error_type, "Short diagnostics must not require an exception allowlist."
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch())
    result = parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)
    assert not result["resolved"] and result["f2p_pass"] == result["p2p_pass"] == 0, (
        "Only a verified task baseline permits the official missing test aggregation."
    )
    assert result["model_failure"]["attribution"] == "baseline_report", "Require independent readiness evidence."


@pytest.mark.parametrize(
    "path",
    [
        "/opt/site-packages/conftest.py",
        "/testbed/.venv/site-packages/conftest.py",
        "/testbed/../conftest.py",
        "/different/conftest.py",
        "/testbed/pandas/unknown.txt",
    ],
)
def test_short_conftest_diagnostic_requires_a_repository_python_path(task, baseline, path):
    text = (
        f"{TEST_OUTPUT_START}\nImportError while loading conftest '{path}'.\n"
        f"ModuleNotFoundError: No module named 'pandas.conftest'\n{TEST_OUTPUT_END}\n"
    )
    assert model_collection_failure_candidate(text, patch()) is None, "External startup diagnostics remain errors."
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)


def test_short_conftest_header_cannot_replace_an_external_exception_frame(task, baseline):
    text = (
        f"{TEST_OUTPUT_START}\nImportError while loading conftest '/testbed/pandas/conftest.py'.\n"
        '  File "/opt/site-packages/dependency.py", line 5, in import_module\n'
        f"ModuleNotFoundError: No module named 'pandas.conftest'\n{TEST_OUTPUT_END}\n"
    )
    assert model_collection_failure_candidate(text, patch()) is None, "Existing terminal frames remain authoritative."
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)


def test_short_conftest_diagnostic_rejects_partial_baseline(task, baseline):
    baseline["test_results_observed"] = 1
    text = (
        f"{TEST_OUTPUT_START}\nImportError while loading conftest '/testbed/pandas/conftest.py'.\n"
        f"ModuleNotFoundError: No module named 'pandas.conftest'\n{TEST_OUTPUT_END}\n"
    )
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)


@pytest.fixture
def plugin_startup_output():
    return (Path(__file__).parent / "fixtures/hydra-1915-plugin-startup.txt").read_text()


def test_pytest_plugin_bootstrap_requires_a_complete_task_baseline(task, baseline, plugin_startup_output):
    model_patch = patch(path="hydra/_internal/utils.py")
    candidate = model_collection_failure_candidate(plugin_startup_output, model_patch)
    assert candidate is not None, "Pytest configuration bootstrap includes plugin loading before conftest imports."
    assert candidate["path"] == "hydra/_internal/utils.py" and candidate["error_type"] == "NameError", (
        "Retain the actual terminal repository exception."
    )
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, plugin_startup_output, "gym", model_patch=model_patch)
    result = parse_benchmark_report(
        task, plugin_startup_output, "gym", model_patch=model_patch, baseline_report=baseline
    )
    assert not result["resolved"] and result["model_failure"]["attribution"] == "baseline_report", (
        "A plugin startup failure requires independently verified task readiness."
    )


def test_plugin_traceback_requires_the_actual_pytest_configuration_bootstrap(task, baseline, plugin_startup_output):
    text = plugin_startup_output.replace("in _prepareconfig", "in ordinary_application")
    assert model_collection_failure_candidate(text, patch()) is None, "Ordinary tracebacks cannot establish startup."
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=patch(), baseline_report=baseline)
