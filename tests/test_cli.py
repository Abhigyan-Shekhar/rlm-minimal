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


def test_interactive_provider_and_model_commands_update_session_state(capsys):
    parser = build_parser()
    cli = RLMCLI(parser.parse_args([]))

    cli._handle_command("/provider gemini")
    cli._handle_command("/model gemini-2.5-flash")

    captured = capsys.readouterr()
    assert cli.args.provider == "gemini"
    assert cli.args.model == "gemini-2.5-flash"
    assert "Provider set to: gemini" in captured.out
    assert "Model set to: gemini-2.5-flash" in captured.out


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


def test_single_shot_missing_api_key_returns_clean_cli_error(monkeypatch, capsys):
    parser = build_parser()
    cli = RLMCLI(parser.parse_args(["--repo", "/tmp/repo", "hello"]))
    monkeypatch.setattr(
        cli,
        "_execute_query",
        lambda query, progress_callback=None: (_ for _ in ()).throw(
            ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        ),
    )

    exit_code = cli.run()

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Configuration error:" in captured.err
    assert "OPENAI_API_KEY" in captured.err
