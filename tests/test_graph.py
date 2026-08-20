"""Tests for LangGraph structure and node functions."""

import unittest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.graph.nodes import evaluator_node, retriever_node
from src.graph.workflow import build_graph


class TestGraphWorkflow(unittest.TestCase):
    def test_compiled_graph_nodes(self) -> None:
        """Verify that the compiled graph contains retriever, response, and evaluator nodes."""
        app = build_graph()
        graph = app.get_graph()
        node_keys = set(graph.nodes.keys())

        # Ensure all three core nodes are present in the graph structure
        self.assertIn("retriever", node_keys)
        self.assertIn("response", node_keys)
        self.assertIn("evaluator", node_keys)

    @patch("src.graph.nodes.read_text_file_via_mcp")
    @patch("src.graph.nodes.retrieve_documents")
    def test_retriever_node_with_mcp_integration(
        self,
        mock_retrieve_docs,
        mock_read_mcp,
    ) -> None:
        """Verify retriever_node retrieves documents and enriches context with MCP content."""
        sample_doc = Document(
            page_content="Core hours are 10 AM to 4 PM.",
            metadata={"source": "remote_work_policy.md"},
        )
        mock_retrieve_docs.return_value = [sample_doc]
        mock_read_mcp.return_value = "# Remote Work Policy Full MCP Content"

        state = {
            "question": "What are the core hours?",
            "documents": [],
            "context": "",
            "mcp_context": "",
            "answer": "",
            "sources": [],
            "evaluation": {},
        }

        result = retriever_node(state)

        self.assertEqual(len(result["documents"]), 1)
        self.assertIn("remote_work_policy.md", result["sources"])
        self.assertEqual(result["mcp_context"], "# Remote Work Policy Full MCP Content")
        self.assertIn("Additional Content via Filesystem MCP Server", result["context"])
        self.assertIn("# Remote Work Policy Full MCP Content", result["context"])
        mock_read_mcp.assert_called_once()

    @patch("src.graph.nodes.evaluate_answer")
    def test_evaluator_node_execution(self, mock_evaluate_answer) -> None:
        """Verify that evaluator_node delegates to evaluate_answer and populates state."""
        mock_evaluate_answer.return_value = {
            "faithfulness": 0.95,
            "answer_relevancy": 0.88,
            "interpretation": "Strong quality",
        }

        dummy_state = {
            "question": "What is the policy?",
            "documents": [],
            "context": "Sample context",
            "mcp_context": "",
            "answer": "Sample answer",
            "sources": ["sample.md"],
            "evaluation": "",
        }

        result = evaluator_node(dummy_state)

        self.assertIn("evaluation", result)
        self.assertEqual(result["evaluation"]["faithfulness"], 0.95)
        self.assertEqual(result["evaluation"]["answer_relevancy"], 0.88)
        self.assertEqual(result["evaluation"]["interpretation"], "Strong quality")


if __name__ == "__main__":
    unittest.main()
