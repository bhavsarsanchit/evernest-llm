import unittest

from evaluate import cited_sources, document_ids, held_back, listing_id


class EvaluateTests(unittest.TestCase):
    def test_listing_id_comes_from_the_question(self):
        self.assertEqual(listing_id("Wie groß ist die Wohnfläche von L-1001?"), "L-1001")

    def test_document_ids_ignore_crm_labels(self):
        self.assertEqual(
            document_ids("D-0102; D-0101; listings.living_area_sqm"),
            (["D-0102", "D-0101"], []),
        )

    def test_a_blocked_document_is_not_required(self):
        self.assertEqual(
            document_ids("D-0109 (darf nicht genutzt werden)"),
            ([], ["D-0109"]),
        )

    def test_cited_sources_is_the_share_of_expected_documents(self):
        score = cited_sources(
            {"expected_sources": "D-0102; D-0101"},
            {"cited": ["D-0102"]},
        )
        self.assertEqual(score.value, 0.5)

    def test_a_blocked_listing_must_be_refused(self):
        score = held_back(
            {"type": "permission", "expected_answer": "Keine Antwort. A-01 hat keinen Zugriff."},
            {"cited": [], "refused": False},
        )
        self.assertEqual(score.value, 0.0)

    def test_crm_questions_do_not_require_a_document(self):
        score = cited_sources(
            {"expected_sources": "offers; activities (CALL zum Angebot)"},
            {"cited": ["D-0201"]},
        )
        self.assertEqual(score.value, 1.0)

    def test_a_cover_agent_must_not_cite_documents(self):
        score = held_back(
            {
                "type": "permission",
                "expected_answer": "A-03 hat per Collaboration Lesezugriff, aber keinen Dokumentzugriff.",
            },
            {"cited": ["D-0106"], "refused": False},
        )
        self.assertEqual(score.value, 0.0)
