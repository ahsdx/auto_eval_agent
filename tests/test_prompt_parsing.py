from auto_eval.judges.prompts import parse_json_loose


def test_parse_json_loose_repairs_unescaped_quotes_in_string_value():
    text = (
        '<analysis>判断正确</analysis>\n'
        '{"rubric":{"准确性":5},"total":5,"correctness":"right",'
        '"error_type":null,"rationale":"下一句"唯见长江天际流"完全正确"}'
    )
    parsed = parse_json_loose(text)
    assert parsed is not None
    assert parsed["correctness"] == "right"
    assert parsed["rationale"] == '下一句"唯见长江天际流"完全正确'


def test_parse_json_loose_removes_trailing_comma():
    parsed = parse_json_loose(
        '{"rubric":{"准确性":5,},"total":5,"correctness":"right",}'
    )
    assert parsed is not None
    assert parsed["correctness"] == "right"