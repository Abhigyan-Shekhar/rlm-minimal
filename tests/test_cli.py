from pathlib import Path

from rlm.cli import RLMCLI, build_parser


def test_file_and_repo_commands_are_mutually_exclusive(tmp_path, capsys):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")
    parser = build_parser()
    cli = RLMCLI(parser.parse_args([]))

    cli._handle_command(f"/file {sample}")
    assert cli.file_context == "hello"
    assert cli.repo_path is None

    cli._handle_command(f"/repo {tmp_path}")
    assert cli.repo_path == str(tmp_path.resolve())
    assert cli.file_context is None


def test_single_shot_returns_non_zero_on_model_failure(monkeypatch, tmp_path, capsys):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")
    parser = build_parser()
    cli = RLMCLI(parser.parse_args(["--file", str(sample), "what", "now?"]))
    monkeypatch.setattr(
        cli,
        "_execute_query",
        lambda query, progress_callback=None: ("MODEL_FAILURE: nope", Path("/tmp/fake.json")),
    )

    exit_code = cli.run()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "MODEL_FAILURE" in captured.out


def test_verbose_progress_goes_to_stderr(capsys):
    parser = build_parser()
    cli = RLMCLI(parser.parse_args(["--verbose", "hello"]))

    callback = cli._make_progress_callback(True)
    callback({"iteration": 2, "event": "code_execution", "code_blocks": 1})

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "iteration 2" in captured.err
