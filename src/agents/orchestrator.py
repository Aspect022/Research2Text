"""
Orchestration layer for managing multi-agent workflow.
Coordinates task dispatch, result aggregation, and error recovery.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import time
import logging

from .base import BaseAgent, AgentMessage, AgentResponse
from .ingest_agent import IngestAgent
from .vision_agent import VisionAgent
from .chunking_agent import ChunkingAgent
from .method_extractor_agent import MethodExtractorAgent
from .equation_agent import EquationAgent
from .dataset_loader_agent import DatasetLoaderAgent
from .code_architect_agent import CodeArchitectAgent
from .graph_builder_agent import GraphBuilderAgent
from .validator_agent import ValidatorAgent
from .cleaner_agent import CleanerAgent

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Central coordinator for the multi-agent system.
    Manages workflow, task dispatch, result aggregation, and error recovery.
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all specialized agents."""
        self.agents["ingest"] = IngestAgent()
        self.agents["vision"] = VisionAgent()
        self.agents["chunking"] = ChunkingAgent()
        self.agents["method_extractor"] = MethodExtractorAgent()
        self.agents["equation"] = EquationAgent()
        self.agents["dataset_loader"] = DatasetLoaderAgent()
        self.agents["code_architect"] = CodeArchitectAgent()
        self.agents["graph_builder"] = GraphBuilderAgent()
        self.agents["validator"] = ValidatorAgent()
        self.agents["cleaner"] = CleanerAgent()
    
    def dispatch(self, agent_id: str, message: AgentMessage) -> AgentResponse:
        """
        Dispatch a message to a specific agent.
        
        Args:
            agent_id: ID of the target agent
            message: Message to send
            
        Returns:
            AgentResponse from the agent
        """
        if agent_id not in self.agents:
            return AgentResponse(
                success=False,
                error=f"Agent '{agent_id}' not found. Available: {list(self.agents.keys())}"
            )
        
        agent = self.agents[agent_id]
        if not agent.validate_message(message):
            return AgentResponse(
                success=False,
                error=f"Message validation failed for agent '{agent_id}'"
            )
        
        start_time = time.time()
        try:
            logger.info(f"[Orchestrator] Dispatching to '{agent_id}'...")
            response = agent.process(message)
            response.processing_time = time.time() - start_time
            logger.info(f"[Orchestrator] '{agent_id}' completed in {response.processing_time:.2f}s (success={response.success})")
            if not response.success:
                logger.warning(f"[Orchestrator] '{agent_id}' failed: {response.error}")
            return response
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Orchestrator] '{agent_id}' raised exception after {elapsed:.2f}s: {e}")
            return AgentResponse(
                success=False,
                error=f"Agent '{agent_id}' raised exception: {str(e)}",
                processing_time=elapsed
            )
    
    def process_paper(self, pdf_path: Optional[Path] = None, paper_base: Optional[str] = None, text: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a research paper through the complete pipeline.
        This is the main entry point for paper processing.
        
        Args:
            pdf_path: Path to the PDF file (optional if text is provided)
            paper_base: Optional base name for the paper
            text: Optional pre-extracted text (if PDF not available)
            
        Returns:
            Dictionary containing all extracted information and generated artifacts
        """
        if pdf_path is None and text is None:
            return {
                "paper_base": paper_base or "unknown",
                "stages": {},
                "errors": ["Either pdf_path or text must be provided"]
            }
        
        if paper_base is None:
            if pdf_path:
                paper_base = pdf_path.stem
            else:
                paper_base = "unknown"
        
        results = {
            "paper_base": paper_base,
            "pdf_path": str(pdf_path) if pdf_path else None,
            "stages": {},
            "errors": []
        }
        
        # ──────────────────────────────────────────────
        # Stage 1: Document Ingestion
        # ──────────────────────────────────────────────
        logger.info(f"[Pipeline] Stage 1: Ingestion for '{paper_base}'")
        if pdf_path and pdf_path.exists():
            ingest_msg = AgentMessage(
                agent_id="orchestrator",
                message_type="request",
                payload={"pdf_path": str(pdf_path), "paper_base": paper_base}
            )
        elif text:
            ingest_msg = AgentMessage(
                agent_id="orchestrator",
                message_type="request",
                payload={"text": text, "paper_base": paper_base}
            )
        else:
            results["errors"].append("No PDF path or text provided")
            return results
        
        ingest_response = self.dispatch("ingest", ingest_msg)
        results["stages"]["ingestion"] = ingest_response.model_dump()
        
        if not ingest_response.success:
            results["errors"].append(f"Ingestion failed: {ingest_response.error}")
            # Try to continue with raw text if available
            if not text:
                return results
            extracted_text = text
        else:
            extracted_text = ingest_response.data.get("text", text or "")
        
        images = ingest_response.data.get("images", [])
        metadata = ingest_response.data.get("metadata", {})
        
        # ──────────────────────────────────────────────
        # Stage 2: Vision Processing (only for images with actual paths)
        # ──────────────────────────────────────────────
        logger.info(f"[Pipeline] Stage 2: Vision Processing ({len(images)} images found)")
        vision_results = []
        for img_info in images:
            img_path = img_info.get("path")
            if not img_path:
                # Skip images without saved paths (not yet extracted to disk)
                continue
            vision_msg = AgentMessage(
                agent_id="orchestrator",
                message_type="request",
                payload={"image_path": img_path, "image_type": img_info.get("type", "unknown")}
            )
            vision_response = self.dispatch("vision", vision_msg)
            vision_results.append(vision_response.model_dump())
        results["stages"]["vision"] = vision_results
        
        # ──────────────────────────────────────────────
        # Stage 3: Chunking
        # ──────────────────────────────────────────────
        logger.info(f"[Pipeline] Stage 3: Chunking ({len(extracted_text)} chars)")
        chunking_msg = AgentMessage(
            agent_id="orchestrator",
            message_type="request",
            payload={"text": extracted_text, "paper_base": paper_base}
        )
        chunking_response = self.dispatch("chunking", chunking_msg)
        results["stages"]["chunking"] = chunking_response.model_dump()
        
        chunks = chunking_response.data.get("chunks", [])
        
        # ──────────────────────────────────────────────
        # Stage 4: Method Extraction
        # ──────────────────────────────────────────────
        logger.info(f"[Pipeline] Stage 4: Method Extraction")
        method_msg = AgentMessage(
            agent_id="orchestrator",
            message_type="request",
            payload={"text": extracted_text, "chunks": chunks}
        )
        method_response = self.dispatch("method_extractor", method_msg)
        results["stages"]["method_extraction"] = method_response.model_dump()
        
        method_struct = method_response.data.get("method_struct", {})
        
        # ──────────────────────────────────────────────
        # Stage 5: Equation Processing
        # ──────────────────────────────────────────────
        equations = method_struct.get("equations", [])
        logger.info(f"[Pipeline] Stage 5: Equation Processing ({len(equations)} equations)")
        equation_results = []
        for eq in equations:
            eq_msg = AgentMessage(
                agent_id="orchestrator",
                message_type="request",
                payload={"equation": eq, "format": "latex"}
            )
            eq_response = self.dispatch("equation", eq_msg)
            equation_results.append(eq_response.model_dump())
        results["stages"]["equations"] = equation_results
        
        # ──────────────────────────────────────────────
        # Stage 6: Dataset Processing
        # ──────────────────────────────────────────────
        datasets = method_struct.get("datasets", [])
        logger.info(f"[Pipeline] Stage 6: Dataset Processing ({len(datasets)} datasets)")
        dataset_msg = AgentMessage(
            agent_id="orchestrator",
            message_type="request",
            payload={"datasets": datasets}
        )
        dataset_response = self.dispatch("dataset_loader", dataset_msg)
        results["stages"]["datasets"] = dataset_response.model_dump()
        
        # ──────────────────────────────────────────────
        # Stage 7: Code Generation
        # ──────────────────────────────────────────────
        logger.info(f"[Pipeline] Stage 7: Code Generation")
        code_msg = AgentMessage(
            agent_id="orchestrator",
            message_type="request",
            payload={
                "method_struct": method_struct,
                "equations": [r.get("data", {}) for r in equation_results if r.get("success")],
                "datasets": dataset_response.data
            }
        )
        code_response = self.dispatch("code_architect", code_msg)
        results["stages"]["code_generation"] = code_response.model_dump()
        
        # ──────────────────────────────────────────────
        # Stage 8: Knowledge Graph Construction
        # ──────────────────────────────────────────────
        logger.info(f"[Pipeline] Stage 8: Knowledge Graph Construction")
        graph_msg = AgentMessage(
            agent_id="orchestrator",
            message_type="request",
            payload={
                "paper_base": paper_base,
                "method_struct": method_struct,
                "chunks": chunks,
                "equations": equation_results,
                "datasets": datasets
            }
        )
        graph_response = self.dispatch("graph_builder", graph_msg)
        results["stages"]["knowledge_graph"] = graph_response.model_dump()
        
        # ──────────────────────────────────────────────
        # Stage 9: Validation
        # ──────────────────────────────────────────────
        generated_files = code_response.data.get("files", [])
        logger.info(f"[Pipeline] Stage 9: Validation ({len(generated_files)} files)")
        if generated_files:
            validation_msg = AgentMessage(
                agent_id="orchestrator",
                message_type="request",
                payload={"files": generated_files}
            )
            validation_response = self.dispatch("validator", validation_msg)
            results["stages"]["validation"] = validation_response.model_dump()
        
        logger.info(f"[Pipeline] Complete for '{paper_base}'. Errors: {len(results['errors'])}")
        return results
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status and capabilities of all agents."""
        return {
            agent_id: {
                "name": agent.agent_name,
                "capabilities": agent.get_capabilities()
            }
            for agent_id, agent in self.agents.items()
        }
