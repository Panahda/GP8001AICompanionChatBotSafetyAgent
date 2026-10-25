"""Adversarial and corner-case stress tests for safety.env."""
import os
import sys
import tempfile
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety.env import load_env, parse_env_line, _find_env_file


class TestParseEnvLineCornerCases:
    """Stress-test parse_env_line with adversarial inputs."""

    def test_empty_and_whitespace(self):
        assert parse_env_line("") is None
        assert parse_env_line("\n") is None
        assert parse_env_line("\r\n") is None
        assert parse_env_line("   ") is None
        assert parse_env_line("\t\t  \r\n") is None

    def test_pure_comments(self):
        assert parse_env_line("#") is None
        assert parse_env_line("# full line comment") is None
        assert parse_env_line("   # leading space comment") is None
        assert parse_env_line("\t# leading tab comment") is None
        assert parse_env_line("###### multiple hashes") is None

    def test_malformed_no_equals(self):
        assert parse_env_line("NO_EQUALS") is None
        assert parse_env_line("export ONLY_KEY") is None
        assert parse_env_line("just some random prose here") is None

    def test_missing_key(self):
        assert parse_env_line("=value") is None
        assert parse_env_line("   = value") is None
        assert parse_env_line("export =value") is None

    def test_empty_values(self):
        assert parse_env_line("EMPTY=") == ("EMPTY", "")
        assert parse_env_line("EMPTY=   ") == ("EMPTY", "")
        assert parse_env_line("  SPACED_KEY  =  ") == ("SPACED_KEY", "")

    def test_multiple_equals(self):
        assert parse_env_line("URL=https://abc.com?x=1&y=2") == ("URL", "https://abc.com?x=1&y=2")
        assert parse_env_line("EQUALS=a=b=c") == ("EQUALS", "a=b=c")
        assert parse_env_line("MULTIPLE_EQ====") == ("MULTIPLE_EQ", "===")
        assert parse_env_line("EQUALS_WITH_SPACES= a = b = c ") == ("EQUALS_WITH_SPACES", "a = b = c")

    def test_export_variations(self):
        assert parse_env_line("export KEY=value") == ("KEY", "value")
        assert parse_env_line("   export KEY=value") == ("KEY", "value")
        assert parse_env_line("export    KEY=value") == ("KEY", "value")
        assert parse_env_line("exportKEY=value") == ("exportKEY", "value")
        assert parse_env_line("export =val") is None

    def test_quoted_values(self):
        assert parse_env_line('DOUBLE="hello world"') == ("DOUBLE", "hello world")
        assert parse_env_line("SINGLE='hello world'") == ("SINGLE", "hello world")
        assert parse_env_line('EMPTY_DOUBLE=""') == ("EMPTY_DOUBLE", "")
        assert parse_env_line("EMPTY_SINGLE=''") == ("EMPTY_SINGLE", "")
        assert parse_env_line('SPACES_OUTSIDE=  "quoted"  ') == ("SPACES_OUTSIDE", "quoted")
        assert parse_env_line('PRESERVE_SPACES="   inner spaces   "') == ("PRESERVE_SPACES", "   inner spaces   ")

    def test_quotes_nested(self):
        assert parse_env_line("JSON='{\"key\": \"value\"}'") == ("JSON", '{"key": "value"}')
        assert parse_env_line('INNER_SINGLE="He said \'hello\'"') == ("INNER_SINGLE", "He said 'hello'")

    def test_mismatched_quotes(self):
        # Mismatched quotes should NOT be stripped as a valid pair
        assert parse_env_line("MISMATCH1=\"hello'") == ("MISMATCH1", "\"hello'")
        assert parse_env_line("MISMATCH2='hello\"") == ("MISMATCH2", "'hello\"")
        assert parse_env_line("SINGLE_QUOTE_ONLY=\"") == ("SINGLE_QUOTE_ONLY", '"')
        assert parse_env_line("SINGLE_APOS_ONLY='") == ("SINGLE_APOS_ONLY", "'")

    def test_inline_comments_unquoted(self):
        assert parse_env_line("KEY=val # inline comment") == ("KEY", "val")
        assert parse_env_line("KEY=val   # lots of spaces") == ("KEY", "val")
        assert parse_env_line("KEY=val#no_space_not_comment") == ("KEY", "val#no_space_not_comment")
        assert parse_env_line("PASSWORD=p#ssw#rd") == ("PASSWORD", "p#ssw#rd")
        assert parse_env_line("COLOR=#FFFFFF") == ("COLOR", "#FFFFFF")

    def test_inline_comments_inside_quotes(self):
        # Comments inside quotes must be preserved as literal characters
        assert parse_env_line('KEY="val # not a comment"') == ("KEY", "val # not a comment")
        assert parse_env_line("KEY='val # not a comment'") == ("KEY", "val # not a comment")

    def test_quoted_values_with_trailing_inline_comment(self):
        # Adversarial case: Quoted value followed by inline comment
        # e.g., KEY="value" # comment
        # Note: In safety/env.py line 54, quote check is evaluated before comment strip,
        # so 'val' at quote check is '"value" # comment', failing quote match!
        res = parse_env_line('KEY="value" # comment')
        assert res is not None
        assert res[0] == "KEY"
        # Trailing comment is stripped and outer quotes are stripped
        assert res[1] == "value"

    def test_tab_separated_inline_comment(self):
        # Inline comment separated by tab
        res = parse_env_line("KEY=value\t# comment")
        # In current implementation, only " #" is stripped, not "\t#"
        assert res is not None
        assert res[0] == "KEY"
        assert res[1] == "value\t# comment"

    def test_unicode_and_special_chars(self):
        assert parse_env_line("GREETING=你好世界") == ("GREETING", "你好世界")
        assert parse_env_line('EMOJI="🤖 CompanionAI"') == ("EMOJI", "🤖 CompanionAI")
        assert parse_env_line(r"PATH_VAL=C:\Users\test\data") == ("PATH_VAL", r"C:\Users\test\data")
        assert parse_env_line("SPECIAL=foo$bar&baz;qux") == ("SPECIAL", "foo$bar&baz;qux")


class TestLoadEnvStress:
    """Stress-test load_env with file variations and system state."""

    def test_nonexistent_file(self):
        assert load_env("nonexistent_file_definitely_not_here.env") is False
        assert load_env(Path("another_nonexistent_path.env")) is False

    def test_directory_path(self):
        # Passing a directory instead of a file should return False without raising
        with tempfile.TemporaryDirectory() as tmpdir:
            assert load_env(tmpdir) is False

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".env") as tmp:
            tmp.write("")
            tmp_path = tmp.name
        try:
            assert load_env(tmp_path) is True
        finally:
            os.unlink(tmp_path)

    def test_whitespace_and_comments_only_file(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".env") as tmp:
            tmp.write("   \n\n# comment 1\n   # comment 2\n\t\n")
            tmp_path = tmp.name
        try:
            assert load_env(tmp_path) is True
        finally:
            os.unlink(tmp_path)

    def test_mixed_valid_and_malformed_lines(self):
        lines = [
            "# Header comment\n",
            "VALID_1=first_value\n",
            "MALFORMED_NO_EQUALS\n",
            "=NO_KEY\n",
            "\n",
            "   # Indented comment\n",
            "export VALID_2=\"second value\"\n",
            "VALID_3=third # inline comment\n",
            "VALID_4=url?a=1&b=2\n",
            "   \n",
        ]
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".env") as tmp:
            tmp.writelines(lines)
            tmp_path = tmp.name

        keys = ["VALID_1", "VALID_2", "VALID_3", "VALID_4"]
        for k in keys:
            os.environ.pop(k, None)

        try:
            assert load_env(tmp_path) is True
            assert os.environ.get("VALID_1") == "first_value"
            assert os.environ.get("VALID_2") == "second value"
            assert os.environ.get("VALID_3") == "third"
            assert os.environ.get("VALID_4") == "url?a=1&b=2"
            assert "MALFORMED_NO_EQUALS" not in os.environ
            assert "" not in os.environ
        finally:
            os.unlink(tmp_path)
            for k in keys:
                os.environ.pop(k, None)

    def test_non_destructive_and_override(self):
        var_name = "STRESS_TEST_NON_DESTRUCTIVE"
        os.environ[var_name] = "initial_system_value"

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".env") as tmp:
            tmp.write(f"{var_name}=from_file\n")
            tmp_path = tmp.name

        try:
            # Default override=False: should NOT overwrite
            assert load_env(tmp_path, override=False) is True
            assert os.environ[var_name] == "initial_system_value"

            # override=True: SHOULD overwrite
            assert load_env(tmp_path, override=True) is True
            assert os.environ[var_name] == "from_file"
        finally:
            os.unlink(tmp_path)
            os.environ.pop(var_name, None)

    def test_utf8_encoding_support(self):
        var_name = "STRESS_TEST_UTF8"
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".env", encoding="utf-8") as tmp:
            tmp.write(f'{var_name}="新加坡 Hawker Centre 🇸🇬"\n')
            tmp_path = tmp.name

        try:
            os.environ.pop(var_name, None)
            assert load_env(tmp_path) is True
            assert os.environ.get(var_name) == "新加坡 Hawker Centre 🇸🇬"
        finally:
            os.unlink(tmp_path)
            os.environ.pop(var_name, None)

    def test_utf8_with_bom_behavior(self):
        # Many Windows text editors (Notepad, PowerShell) output UTF-8 with BOM (\xef\xbb\xbf)
        # Because safety/env.py opens with encoding="utf-8" instead of "utf-8-sig",
        # the leading key gets prefixed with \ufeff.
        var_name = "TEST_BOM_KEY"
        with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".env") as tmp:
            # Write UTF-8 BOM followed by key=val
            tmp.write(b"\xef\xbb\xbfTEST_BOM_KEY=bom_value\n")
            tmp_path = tmp.name

        try:
            os.environ.pop(var_name, None)
            os.environ.pop("\ufeff" + var_name, None)
            assert load_env(tmp_path) is True
            # With encoding="utf-8-sig", key is parsed without BOM prefix
            has_clean_key = var_name in os.environ
            has_bom_key = ("\ufeff" + var_name) in os.environ
            assert has_clean_key
            assert not has_bom_key
            assert os.environ.get(var_name) == "bom_value"
        finally:
            os.unlink(tmp_path)
            os.environ.pop(var_name, None)
            os.environ.pop("\ufeff" + var_name, None)

    def test_duplicate_keys_in_file(self):
        # When duplicate keys exist in the same file:
        # With override=False, the first occurrence takes precedence because it is inserted into os.environ
        # and subsequent lines see key in os.environ.
        var_name = "STRESS_DUP_KEY"
        os.environ.pop(var_name, None)
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".env") as tmp:
            tmp.write(f"{var_name}=first\n{var_name}=second\n")
            tmp_path = tmp.name

        try:
            assert load_env(tmp_path, override=False) is True
            assert os.environ.get(var_name) == "first"

            # With override=True, later occurrence overwrites earlier occurrence
            assert load_env(tmp_path, override=True) is True
            assert os.environ.get(var_name) == "second"
        finally:
            os.unlink(tmp_path)
            os.environ.pop(var_name, None)

    def test_large_file_stress(self):
        # Stress test loading a file with 1000 variables
        count = 1000
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".env") as tmp:
            for i in range(count):
                tmp.write(f"STRESS_VAR_{i}=value_{i}\n")
            tmp_path = tmp.name

        try:
            assert load_env(tmp_path) is True
            assert os.environ.get("STRESS_VAR_0") == "value_0"
            assert os.environ.get(f"STRESS_VAR_{count-1}") == f"value_{count-1}"
        finally:
            os.unlink(tmp_path)
            for i in range(count):
                os.environ.pop(f"STRESS_VAR_{i}", None)

    def test_find_env_file_behavior(self):
        # Test _find_env_file finds existing .env
        found = _find_env_file()
        # In this repository, .env exists at project root
        assert found is not None
        assert found.name == ".env"
        assert found.is_file()
