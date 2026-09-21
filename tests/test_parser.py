from muyah_code.llm.client import parse_arguments, split_think
from muyah_code.llm.toolcall_parser import parse_tool_calls
from muyah_code.tools import default_registry

REG = default_registry()


def parse(text):
    return parse_tool_calls(text, REG.resolve)


def test_hermes_tool_call_tag():
    calls, rest = parse('Let me look.\n<tool_call>\n{"name": "Read", "arguments": {"file_path": "a.py"}}\n</tool_call>')
    assert [(c.name, c.arguments) for c in calls] == [("Read", {"file_path": "a.py"})]
    assert rest == "Let me look."


def test_multiple_and_unclosed_tags():
    text = ('<tool_call>{"name": "Glob", "arguments": {"pattern": "*.py"}}</tool_call>\n'
            '<tool_call>{"name": "Grep", "arguments": {"pattern": "def main"}}')
    calls, _ = parse(text)
    assert [c.name for c in calls] == ["Glob", "Grep"]


def test_qwen3_coder_xml_format():
    text = ("<tool_call>\n<function=Bash>\n<parameter=command>\npytest -q\n</parameter>\n"
            "<parameter=timeout>\n60\n</parameter>\n</function>\n</tool_call>")
    calls, _ = parse(text)
    assert calls[0].name == "Bash"
    assert calls[0].arguments == {"command": "pytest -q", "timeout": 60}


def test_aliases_resolve_to_canonical_names():
    calls, _ = parse('<tool_call>{"name": "write_file", "arguments": {"file_path": "x", "content": "y"}}</tool_call>')
    assert calls[0].name == "Write"


def test_raw_newlines_inside_json_strings_are_tolerated():
    text = '<tool_call>{"name": "Write", "arguments": {"file_path": "a.py", "content": "line1\nline2"}}</tool_call>'
    calls, _ = parse(text)
    assert calls[0].arguments["content"] == "line1\nline2"


def test_trailing_commas_are_repaired():
    calls, _ = parse('<tool_call>{"name": "Read", "arguments": {"file_path": "a.py",},}</tool_call>')
    assert calls[0].arguments == {"file_path": "a.py"}


def test_code_blocks_never_become_writes():
    """The old agent turned these into write_file calls and littered projects with junk."""
    text = ("Here is the file:\n\n```python:src/app.py\nprint('hi')\n```\n\n"
            "File: notes.md\n```markdown\n# notes\n```\n")
    calls, rest = parse(text)
    assert calls == []
    assert rest == text.strip()


def test_json_example_without_arguments_key_is_not_a_call():
    calls, _ = parse('The config looks like {"name": "Read", "value": 3}.')
    assert calls == []


def test_bare_json_with_arguments_key_is_a_call():
    calls, _ = parse('{"name": "LS", "arguments": {"path": "."}}')
    assert calls[0].name == "LS"


def test_unknown_tool_names_are_ignored_in_bare_json():
    calls, _ = parse('{"name": "launch_missiles", "arguments": {}}')
    assert calls == []


def test_unparseable_tool_call_tag_surfaces_an_error():
    calls, _ = parse("<tool_call>{this is not json}</tool_call>")
    assert calls and calls[0].parse_error


def test_fenced_tool_call_block():
    calls, _ = parse('```tool_call\n{"name": "Read", "arguments": {"file_path": "b.py"}}\n```')
    assert calls[0].arguments == {"file_path": "b.py"}


def test_parse_arguments_variants():
    assert parse_arguments('{"a": 1}') == ({"a": 1}, None)
    assert parse_arguments('"{\\"a\\": 1}"') == ({"a": 1}, None)  # double-encoded
    assert parse_arguments("") == ({}, None)
    args, err = parse_arguments("[1,2]")
    assert args is None and err


def test_split_think():
    visible, reasoning = split_think("<think>plan it</think>The answer.")
    assert visible == "The answer." and reasoning == "plan it"
