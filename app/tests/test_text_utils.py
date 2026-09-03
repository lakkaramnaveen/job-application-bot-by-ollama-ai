from job_bot.text_utils import normalize_company_name


def test_normalize_company_name_strips_and_casefolds():
    assert normalize_company_name("  Acme Corp  ") == "acme corp"
    assert normalize_company_name("ACME CORP") == "acme corp"


def test_normalize_company_name_collapses_internal_whitespace():
    assert normalize_company_name("Acme   Corp") == "acme corp"
    assert normalize_company_name("Acme\tCorp\n") == "acme corp"
