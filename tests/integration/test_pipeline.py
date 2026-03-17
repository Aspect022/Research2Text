"""Integration tests for the 3-phase pipeline."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch


class TestResearchPhase:
    """Test cases for Phase 1: Research Phase."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        from agents.orchestrator import Orchestrator
        orchestrator = Orchestrator()

        # Mock agent responses
        orchestrator.agents["ingest"] = Mock()
        orchestrator.agents["ingest"].process.return_value = Mock(
            success=True,
            data={
                "text": "Sample paper text",
                "images": [],
                "tables": [],
                "equations": [],
                "metadata": {}
            },
            error=None,
            processing_time=1.0
        )

        orchestrator.agents["vision"] = Mock()
        orchestrator.agents["vision"].process.return_value = Mock(
            success=True, data={}, error=None, processing_time=0.5
        )

        orchestrator.agents["chunking"] = Mock()
        orchestrator.agents["chunking"].process.return_value = Mock(
            success=True, data={"chunks": ["chunk1", "chunk2"]}, error=None, processing_time=0.5
        )

        orchestrator.agents["method_extractor"] = Mock()
        orchestrator.agents["method_extractor"].process.return_value = Mock(
            success=True,
            data={
                "method_struct": {
                    "algorithm_name": "Test Algorithm",
                    "datasets": ["CIFAR-10"],
                    "equations": []
                },
                "overall_confidence": 0.85
            },
            error=None,
            processing_time=2.0
        )

        orchestrator.agents["equation"] = Mock()
        orchestrator.agents["equation"].process.return_value = Mock(
            success=True, data={}, error=None, processing_time=0.5
        )

        orchestrator.agents["dataset_loader"] = Mock()
        orchestrator.agents["dataset_loader"].process.return_value = Mock(
            success=True, data={"datasets": {}}, error=None, processing_time=0.5
        )

        orchestrator.agents["graph_builder"] = Mock()
        orchestrator.agents["graph_builder"].process.return_value = Mock(
            success=True, data={"nodes": [], "edges": []}, error=None, processing_time=1.0
        )

        return orchestrator

    def test_process_paper_to_knowledge_graph(self, mock_orchestrator):
        """Test the full research phase."""
        result = mock_orchestrator.process_paper_to_knowledge_graph(
            text="Sample paper text",
            paper_base="test_paper"
        )

        assert result["paper_base"] == "test_paper"
        assert result["ready_for_code_gen"] == True
        assert "method_struct" in result
        assert "knowledge_graph" in result
        assert len(result["errors"]) == 0

    def test_research_phase_with_errors(self, mock_orchestrator):
        """Test research phase with agent errors."""
        # Make method extractor fail
        mock_orchestrator.agents["method_extractor"].process.return_value = Mock(
            success=False, data={}, error="Extraction failed", processing_time=0.0
        )

        result = mock_orchestrator.process_paper_to_knowledge_graph(
            text="Sample paper text",
            paper_base="test_paper"
        )

        assert result["ready_for_code_gen"] == False
        assert len(result["errors"]) > 0


class TestCodeGenerationPhase:
    """Test cases for Phase 2: Code Generation."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        from agents.orchestrator import Orchestrator
        orchestrator = Orchestrator()

        orchestrator.agents["code_architect"] = Mock()
        orchestrator.agents["code_architect"].process.return_value = Mock(
            success=True,
            data={
                "files": [
                    {"path": "model.py", "content": "class Model: pass"},
                    {"path": "train.py", "content": "# Training code"}
                ]
            },
            error=None,
            processing_time=3.0
        )

        return orchestrator

    def test_generate_code(self, mock_orchestrator):
        """Test code generation."""
        method_struct = {
            "algorithm_name": "Test",
            "architecture": {"layer_types": ["Conv2D", "Dense"]}
        }

        result = mock_orchestrator.generate_code(
            paper_base="test_paper",
            method_struct=method_struct,
            equations=[],
            datasets={},
            paper_text="Sample text"
        )

        assert result["stage"] == "code_generation"
        assert result["success"] == True
        assert len(result["files"]) == 2

    def test_generate_code_failure(self, mock_orchestrator):
        """Test code generation failure."""
        mock_orchestrator.agents["code_architect"].process.return_value = Mock(
            success=False, data={}, error="Generation failed", processing_time=0.0
        )

        result = mock_orchestrator.generate_code(
            paper_base="test_paper",
            method_struct={},
            equations=[],
            datasets={},
            paper_text=""
        )

        assert result["success"] == False
        assert result["error"] == "Generation failed"


class TestSandboxPhase:
    """Test cases for Phase 3: Sandbox Execution."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        from agents.orchestrator import Orchestrator
        orchestrator = Orchestrator()

        orchestrator.agents["validator"] = Mock()
        orchestrator.agents["validator"].process.return_value = Mock(
            success=True,
            data={
                "syntax_correctness": 1.0,
                "import_resolution": 1.0,
                "execution": {
                    "success": True,
                    "attempts": 1,
                    "stdout": "Training complete",
                    "stderr": ""
                }
            },
            error=None,
            processing_time=10.0
        )

        return orchestrator

    def test_run_sandbox_validation(self, mock_orchestrator):
        """Test sandbox validation."""
        files = [
            {"path": "model.py", "content": "print('hello')"}
        ]

        result = mock_orchestrator.run_sandbox_validation(
            paper_base="test_paper",
            files=files
        )

        assert result["stage"] == "validation"
        assert result["success"] == True
        assert "validation" in result

    def test_run_sandbox_no_files(self, mock_orchestrator):
        """Test sandbox with no files."""
        result = mock_orchestrator.run_sandbox_validation(
            paper_base="test_paper",
            files=[]
        )

        assert result["success"] == False
        assert "No files to validate" in result["error"]


class TestEndToEndWorkflow:
    """Test cases for end-to-end workflow."""

    def test_full_workflow(self, tmp_path):
        """Test the complete 3-phase workflow."""
        # This is a simplified test - in reality would need more setup
        from agents.orchestrator import Orchestrator

        orchestrator = Orchestrator()

        # Mock all agents
        for agent_id in orchestrator.agents:
            orchestrator.agents[agent_id] = Mock()
            orchestrator.agents[agent_id].process.return_value = Mock(
                success=True,
                data={"files": [{"path": "test.py", "content": "print('test')"}]}
                if agent_id == "code_architect"
                else {"nodes": [], "edges": []}
                if agent_id == "graph_builder"
                else {"method_struct": {"algorithm_name": "Test"}}
                if agent_id == "method_extractor"
                else {},
                error=None,
                processing_time=1.0
            )

        # Phase 1
        result1 = orchestrator.process_paper_to_knowledge_graph(
            text="Sample text",
            paper_base="test"
        )
        assert result1["ready_for_code_gen"]

        # Phase 2
        result2 = orchestrator.generate_code(
            paper_base="test",
            method_struct=result1["method_struct"],
            equations=[],
            datasets={},
            paper_text="Sample text"
        )
        assert result2["success"]

        # Phase 3
        result3 = orchestrator.run_sandbox_validation(
            paper_base="test",
            files=result2["files"]
        )
        assert result3["success"]
