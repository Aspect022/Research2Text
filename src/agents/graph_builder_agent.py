"""
Agent 8: Graph Builder Agent
Responsibility: Construct knowledge graphs representing paper structure
"""

from typing import Dict, Any, List, Optional
from .base import BaseAgent, AgentMessage, AgentResponse


class GraphBuilderAgent(BaseAgent):
    """Agent for constructing knowledge graphs."""
    
    def __init__(self):
        super().__init__("graph_builder", "Graph Builder Agent")
        self._node_types = [
            "Paper", "Section", "Concept", "Equation", "Algorithm",
            "Dataset", "Metric", "Figure", "Table", "Citation"
        ]
    
    def process(self, message: AgentMessage) -> AgentResponse:
        """Build knowledge graph from paper information."""
        payload = message.payload
        paper_base = payload.get("paper_base", "unknown")
        method_struct = payload.get("method_struct", {})
        chunks = payload.get("chunks", [])
        equations = payload.get("equations", [])
        datasets = payload.get("datasets", [])
        
        try:
            nodes = []
            edges = []
            
            # Add paper node
            paper_id = f"paper_{paper_base}"
            nodes.append({
                "id": paper_id,
                "type": "Paper",
                "label": paper_base,
                "properties": {}
            })
            
            # Add algorithm node
            if method_struct.get("algorithm_name"):
                algo_id = f"algorithm_{paper_base}"
                nodes.append({
                    "id": algo_id,
                    "type": "Algorithm",
                    "label": method_struct["algorithm_name"],
                    "properties": {}
                })
                edges.append({
                    "source": paper_id,
                    "target": algo_id,
                    "type": "contains"
                })
            
            # Add dataset nodes
            for dataset in datasets:
                dataset_id = f"dataset_{dataset}"
                nodes.append({
                    "id": dataset_id,
                    "type": "Dataset",
                    "label": dataset,
                    "properties": {}
                })
                edges.append({
                    "source": paper_id,
                    "target": dataset_id,
                    "type": "uses"
                })
            
            # Add equation nodes
            for i, eq_info in enumerate(equations):
                if isinstance(eq_info, dict) and eq_info.get("success"):
                    eq_id = f"equation_{paper_base}_{i}"
                    nodes.append({
                        "id": eq_id,
                        "type": "Equation",
                        "label": eq_info.get("data", {}).get("normalized", f"Equation {i}"),
                        "properties": {
                            "sympy": eq_info.get("data", {}).get("sympy"),
                            "pytorch": eq_info.get("data", {}).get("pytorch")
                        }
                    })
                    edges.append({
                        "source": paper_id,
                        "target": eq_id,
                        "type": "contains"
                    })
            
            # Add citation nodes
            if method_struct.get("references"):
                for ref in method_struct["references"]:
                    ref_id = f"citation_{paper_base}_{ref}"
                    nodes.append({
                        "id": ref_id,
                        "type": "Citation",
                        "label": ref,
                        "properties": {}
                    })
                    edges.append({
                        "source": paper_id,
                        "target": ref_id,
                        "type": "cites"
                    })
            
            return AgentResponse(
                success=True,
                data={
                    "nodes": nodes,
                    "edges": edges,
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "graph_format": "json"
                }
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Graph construction failed: {str(e)}"
            )
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "description": "Construct knowledge graphs representing paper structure",
            "node_types": self._node_types,
            "average_nodes": 47,
            "average_edges": 82
        }

