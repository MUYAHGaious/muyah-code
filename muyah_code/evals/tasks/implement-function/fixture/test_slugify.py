from slugify import slugify

def test_basic():
    assert slugify("Hello World") == "hello-world"

def test_punctuation_and_spaces():
    assert slugify("  Hello,   World!  ") == "hello-world"

def test_numbers_and_dashes():
    assert slugify("Python 3.12 -- Release Notes") == "python-3-12-release-notes"

def test_empty():
    assert slugify("!!!") == ""
