"""Tests for RAGAS evaluation module, score interpretation, and reporting."""

import unittest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.evaluation import evaluate_answer, interpret_scores
from src.run_graph import format_evaluation_report


class TestEvaluation(unittest.TestCase):
    def test_interpret_scores_strong(self) -> None:
        """Scores >= 0.80 should indicate strong quality."""
        result = interpret_scores(0.95, 0.85)
        self.assertIn("Strong quality", result)

        result_boundary = interpret_scores(0.80, 0.80)
        self.assertIn("Strong quality", result_boundary)

    def test_interpret_scores_acceptable(self) -> None:
        """Scores between 0.60 and 0.79 should indicate acceptable quality."""
        result = interpret_scores(0.75, 0.65)
        self.assertIn("Acceptable quality", result)

        result_boundary = interpret_scores(0.60, 0.60)
        self.assertIn("Acceptable quality", result_boundary)

    def test_interpret_scores_needs_review(self) -> None:
        """Scores < 0.60 should indicate needs review."""
        result = interpret_scores(0.50, 0.90)
        self.assertIn("Needs review", result)

        result_low = interpret_scores(0.40, 0.30)
        self.assertIn("Needs review", result_low)

    def test_format_evaluation_report(self) -> None:
        """Verify percentage formatting to two decimal places."""
        eval_dict = {
            "faithfulness": 0.9235,
            "answer_relevancy": 0.8179,
            "interpretation": "Strong quality",
        }
        report = format_evaluation_report(eval_dict)

        self.assertIn("Faithfulness:      92.35%", report)
        self.assertIn("Answer Relevancy:  81.79%", report)
        self.assertIn("Summary:           Strong quality", report)

    def test_evaluate_answer_empty_documents(self) -> None:
        """Empty documents should return 0 scores and needs review without calling external APIs."""
        settings = MagicMock()
        settings.google_api_key = "test-key"

        result = evaluate_answer(
            question="What is the policy?",
            answer="Some answer",
            documents=[],
            settings=settings,
        )

        self.assertEqual(result["faithfulness"], 0.0)
        self.assertEqual(result["answer_relevancy"], 0.0)
        self.assertIn("Needs review", result["interpretation"])

    @patch("src.evaluation.create_chat_model")
    @patch("src.evaluation.get_embedding_function")
    @patch("src.evaluation.evaluate")
    def test_evaluate_answer_mocked_ragas(
        self,
        mock_ragas_evaluate,
        mock_get_embedding,
        mock_create_chat,
    ) -> None:
        """Verify that evaluate_answer correctly extracts and returns RAGAS metric floats."""
        mock_create_chat.return_value = MagicMock()
        mock_get_embedding.return_value = MagicMock()
        mock_ragas_evaluate.return_value = {
            "faithfulness": [0.95],
            "answer_relevancy": [0.88],
        }

        settings = MagicMock()
        settings.google_api_key = "test-key"

        docs = [Document(page_content="Policy content for test.")]
        result = evaluate_answer(
            question="What is the policy?",
            answer="Policy answer",
            documents=docs,
            settings=settings,
        )

        self.assertEqual(result["faithfulness"], 0.95)
        self.assertEqual(result["answer_relevancy"], 0.88)
        self.assertIn("Strong quality", result["interpretation"])


if __name__ == "__main__":
    unittest.main()
