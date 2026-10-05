from pathlib import Path

from app.rag.ingest import chunk_document, load_corpus, load_source
from app.rag.store import build_index, search


def test_corpus_includes_html_and_txt():
    documents = load_corpus()
    formats = {document["format"] for document in documents}
    assert "html" in formats
    assert "txt" in formats
    assert any(document["document_id"] == "200" for document in documents)
    assert any(document["document_id"] == "010" for document in documents)
    sources = {document["source"] for document in documents}
    assert "design-and-evaluation.md" not in sources
    assert "README.md" not in sources


def test_txt_parser_and_heading_chunks(tmp_path: Path):
    path = tmp_path / "250-sample.txt"
    path.write_text(
        "# Sample\n\n## Leave\n\nEmployees receive ten days of paid leave each calendar year.\n\n"
        "## Equipment\n\nLaptops are issued on day one and must be returned.\n"
    )
    document = load_source(path)
    chunks = chunk_document(document)
    assert document["format"] == "txt"
    assert {chunk["section"] for chunk in chunks} >= {"Leave", "Equipment"}
    assert all(chunk["document_id"] == "250" for chunk in chunks)


def test_retrieval_hits_expected_policies():
    build_index()
    expectations = {
        "parental leave 12 weeks": "200",
        "401k match retirement": "520",
        "healthcare medical plan deductible": "510",
        "VPN GlobalProtect company devices remote work data security": "600",
        "home office stipend ergonomic chair": "400",
        "change of work location another state": "300",
        "onboarding day 1 buddy": "700",
        "return company equipment five business days": "800",
        "CorporateTravel hotel per diem": "900",
        "harassment ethics hotline retaliation": "100",
    }
    for query, document_id in expectations.items():
        hits = search(query, k=4)
        found = [hit["document_id"] for hit in hits]
        assert document_id in found, (query, found)
