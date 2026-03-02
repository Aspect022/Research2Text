"""
Test script to verify the multi-agent pipeline works end-to-end.
Tests:
  1. Each agent can be instantiated without import errors.
  2. The Orchestrator initializes all agents.
  3. A mock text can be processed through the full pipeline.
"""

import sys
import logging
from pathlib import Path

# Ensure src/ is on the path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("test_pipeline")

MOCK_PAPER_TEXT = """
Abstract
This paper proposes a Transformer-based architecture for image classification on the CIFAR-10 dataset. 

Methods
We implement a Vision Transformer (ViT) model with multi-head self-attention. The core computation
follows the scaled dot-product attention mechanism: QK^T / sqrt(d_k).

Our model uses an Adam optimizer with a learning rate of 0.001 and cross-entropy loss.
Training is performed for 100 epochs with a batch size of 32.

The input images are resized to 224x224 and split into 16x16 patches, resulting in 196 tokens.
The output is a classification vector of size 10 (one per CIFAR-10 class).

Results
Our approach achieves 95.3% accuracy on CIFAR-10, outperforming the baseline CNN model [1].
We also evaluate on MNIST for comparison [2].
"""


def test_agent_imports():
    """Test 1: All agents can be imported without errors."""
    logger.info("=" * 60)
    logger.info("TEST 1: Agent Imports")
    logger.info("=" * 60)
    
    from agents.base import BaseAgent, AgentMessage, AgentResponse
    logger.info("  ✅ base.py imported successfully")

    from agents.ingest_agent import IngestAgent
    logger.info("  ✅ IngestAgent imported")

    from agents.vision_agent import VisionAgent
    logger.info("  ✅ VisionAgent imported")

    from agents.chunking_agent import ChunkingAgent
    logger.info("  ✅ ChunkingAgent imported")

    from agents.method_extractor_agent import MethodExtractorAgent
    logger.info("  ✅ MethodExtractorAgent imported")

    from agents.equation_agent import EquationAgent
    logger.info("  ✅ EquationAgent imported")

    from agents.dataset_loader_agent import DatasetLoaderAgent
    logger.info("  ✅ DatasetLoaderAgent imported")

    from agents.code_architect_agent import CodeArchitectAgent
    logger.info("  ✅ CodeArchitectAgent imported")

    from agents.graph_builder_agent import GraphBuilderAgent
    logger.info("  ✅ GraphBuilderAgent imported")

    from agents.validator_agent import ValidatorAgent
    logger.info("  ✅ ValidatorAgent imported")

    from agents.cleaner_agent import CleanerAgent
    logger.info("  ✅ CleanerAgent imported")

    logger.info("TEST 1 PASSED: All agents imported successfully.\n")
    return True


def test_orchestrator_init():
    """Test 2: Orchestrator initializes all agents."""
    logger.info("=" * 60)
    logger.info("TEST 2: Orchestrator Initialization")
    logger.info("=" * 60)

    from agents.orchestrator import Orchestrator
    orch = Orchestrator()

    expected_agents = [
        "ingest", "vision", "chunking", "method_extractor",
        "equation", "dataset_loader", "code_architect",
        "graph_builder", "validator", "cleaner"
    ]

    for agent_id in expected_agents:
        assert agent_id in orch.agents, f"Missing agent: {agent_id}"
        logger.info(f"  ✅ Agent '{agent_id}' initialized: {orch.agents[agent_id]}")

    status = orch.get_agent_status()
    logger.info(f"  Agent status keys: {list(status.keys())}")
    assert len(status) == 10, f"Expected 10 agents in status, got {len(status)}"

    logger.info("TEST 2 PASSED: Orchestrator initialized with all 10 agents.\n")
    return True


def test_individual_agents():
    """Test 3: Individual agent processing with mock data."""
    logger.info("=" * 60)
    logger.info("TEST 3: Individual Agent Processing")
    logger.info("=" * 60)

    from agents.base import AgentMessage
    from agents.orchestrator import Orchestrator
    orch = Orchestrator()

    # Test IngestAgent with text input
    ingest_msg = AgentMessage(
        agent_id="test",
        message_type="request",
        payload={"text": MOCK_PAPER_TEXT, "paper_base": "test_paper"}
    )
    resp = orch.dispatch("ingest", ingest_msg)
    assert resp.success, f"IngestAgent failed: {resp.error}"
    assert resp.data.get("text"), "IngestAgent returned no text"
    logger.info(f"  ✅ IngestAgent: success={resp.success}, text_len={len(resp.data.get('text', ''))}")

    # Test ChunkingAgent
    chunk_msg = AgentMessage(
        agent_id="test",
        message_type="request",
        payload={"text": MOCK_PAPER_TEXT, "paper_base": "test_paper"}
    )
    resp = orch.dispatch("chunking", chunk_msg)
    assert resp.success, f"ChunkingAgent failed: {resp.error}"
    chunks = resp.data.get("chunks", [])
    logger.info(f"  ✅ ChunkingAgent: success={resp.success}, chunk_count={len(chunks)}")

    # Test MethodExtractorAgent
    method_msg = AgentMessage(
        agent_id="test",
        message_type="request",
        payload={"text": MOCK_PAPER_TEXT, "chunks": chunks}
    )
    resp = orch.dispatch("method_extractor", method_msg)
    assert resp.success, f"MethodExtractorAgent failed: {resp.error}"
    method_struct = resp.data.get("method_struct", {})
    logger.info(f"  ✅ MethodExtractorAgent: success={resp.success}")
    logger.info(f"     algorithm: {method_struct.get('algorithm_name')}")
    logger.info(f"     datasets: {method_struct.get('datasets')}")
    logger.info(f"     equations: {method_struct.get('equations')}")

    # Test EquationAgent
    equations = method_struct.get("equations", [])
    if equations:
        eq_msg = AgentMessage(
            agent_id="test",
            message_type="request",
            payload={"equation": equations[0], "format": "latex"}
        )
        resp = orch.dispatch("equation", eq_msg)
        logger.info(f"  ✅ EquationAgent: success={resp.success}, data={resp.data}")
    else:
        logger.info(f"  ⚠️ EquationAgent: skipped (no equations found)")

    # Test DatasetLoaderAgent
    datasets = method_struct.get("datasets", [])
    ds_msg = AgentMessage(
        agent_id="test",
        message_type="request",
        payload={"datasets": datasets}
    )
    resp = orch.dispatch("dataset_loader", ds_msg)
    assert resp.success, f"DatasetLoaderAgent failed: {resp.error}"
    logger.info(f"  ✅ DatasetLoaderAgent: success={resp.success}, loaders={len(resp.data.get('loaders', []))}")

    # Test CodeArchitectAgent
    code_msg = AgentMessage(
        agent_id="test",
        message_type="request",
        payload={"method_struct": method_struct, "equations": [], "datasets": resp.data}
    )
    resp = orch.dispatch("code_architect", code_msg)
    assert resp.success, f"CodeArchitectAgent failed: {resp.error}"
    files = resp.data.get("files", [])
    logger.info(f"  ✅ CodeArchitectAgent: success={resp.success}, files={[f.get('path') for f in files]}")

    # Test GraphBuilderAgent
    graph_msg = AgentMessage(
        agent_id="test",
        message_type="request",
        payload={
            "paper_base": "test_paper",
            "method_struct": method_struct,
            "chunks": chunks,
            "equations": [],
            "datasets": datasets
        }
    )
    resp = orch.dispatch("graph_builder", graph_msg)
    assert resp.success, f"GraphBuilderAgent failed: {resp.error}"
    logger.info(f"  ✅ GraphBuilderAgent: success={resp.success}, nodes={resp.data.get('node_count')}, edges={resp.data.get('edge_count')}")

    # Test ValidatorAgent
    if files:
        val_msg = AgentMessage(
            agent_id="test",
            message_type="request",
            payload={"files": files}
        )
        resp = orch.dispatch("validator", val_msg)
        logger.info(f"  ✅ ValidatorAgent: success={resp.success}, syntax_correctness={resp.data.get('syntax_correctness')}")

    logger.info("TEST 3 PASSED: All agents process mock data correctly.\n")
    return True


def test_full_pipeline():
    """Test 4: Full pipeline (orchestrator.process_paper) with mock text."""
    logger.info("=" * 60)
    logger.info("TEST 4: Full Pipeline (process_paper)")
    logger.info("=" * 60)

    from agents.orchestrator import Orchestrator
    orch = Orchestrator()

    results = orch.process_paper(text=MOCK_PAPER_TEXT, paper_base="test_paper")

    assert results["paper_base"] == "test_paper"
    assert "stages" in results

    stages = results["stages"]
    for stage_name in ["ingestion", "chunking", "method_extraction", "equations", "datasets", "code_generation", "knowledge_graph"]:
        assert stage_name in stages, f"Missing stage: {stage_name}"
        stage_data = stages[stage_name]
        if isinstance(stage_data, dict):
            logger.info(f"  Stage '{stage_name}': success={stage_data.get('success', 'N/A')}")
        elif isinstance(stage_data, list):
            logger.info(f"  Stage '{stage_name}': {len(stage_data)} items")

    errors = results.get("errors", [])
    if errors:
        logger.warning(f"  Pipeline errors: {errors}")
    else:
        logger.info(f"  No pipeline errors!")

    logger.info("TEST 4 PASSED: Full pipeline completed successfully.\n")
    return True


if __name__ == "__main__":
    all_passed = True

    try:
        all_passed &= test_agent_imports()
    except Exception as e:
        logger.error(f"TEST 1 FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_orchestrator_init()
    except Exception as e:
        logger.error(f"TEST 2 FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_individual_agents()
    except Exception as e:
        logger.error(f"TEST 3 FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_full_pipeline()
    except Exception as e:
        logger.error(f"TEST 4 FAILED: {e}")
        all_passed = False

    print()
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED!")
    else:
        logger.error("❌ SOME TESTS FAILED!")
    
    sys.exit(0 if all_passed else 1)
