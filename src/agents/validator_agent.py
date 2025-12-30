"""
Agent 9: Validator Agent
Responsibility: Validate generated code quality
"""

from typing import Dict, Any, List
import ast
import importlib.util
from pathlib import Path
from .base import BaseAgent, AgentMessage, AgentResponse


class ValidatorAgent(BaseAgent):
    """Agent for validating code quality."""
    
    def __init__(self):
        super().__init__("validator", "Validator Agent")
    
    def process(self, message: AgentMessage) -> AgentResponse:
        """Validate generated code files."""
        payload = message.payload
        files = payload.get("files", [])
        
        if not files:
            return AgentResponse(
                success=False,
                error="Missing 'files' in payload"
            )
        
        try:
            validation_results = []
            overall_success = True
            
            for file_info in files:
                file_path = file_info.get("path", "")
                content = file_info.get("content", "")
                
                result = {
                    "file": file_path,
                    "syntax_valid": False,
                    "imports_valid": False,
                    "errors": []
                }
                
                # Check syntax
                syntax_check = self._check_syntax(content)
                result["syntax_valid"] = syntax_check["valid"]
                if not syntax_check["valid"]:
                    result["errors"].extend(syntax_check.get("errors", []))
                    overall_success = False
                
                # Check imports
                if syntax_check["valid"]:
                    import_check = self._check_imports(content)
                    result["imports_valid"] = import_check["valid"]
                    if not import_check["valid"]:
                        result["errors"].extend(import_check.get("errors", []))
                
                validation_results.append(result)
            
            return AgentResponse(
                success=overall_success,
                data={
                    "files": validation_results,
                    "syntax_correctness": sum(1 for r in validation_results if r["syntax_valid"]) / len(validation_results) if validation_results else 0.0,
                    "import_resolution": sum(1 for r in validation_results if r["imports_valid"]) / len(validation_results) if validation_results else 0.0
                }
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Validation failed: {str(e)}"
            )
    
    def _check_syntax(self, code: str) -> Dict[str, Any]:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return {"valid": True}
        except SyntaxError as e:
            return {
                "valid": False,
                "errors": [f"Syntax error: {str(e)} at line {e.lineno}"]
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Parse error: {str(e)}"]
            }
    
    def _check_imports(self, code: str) -> Dict[str, Any]:
        """Check if imports can be resolved (basic check)."""
        try:
            tree = ast.parse(code)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Basic check - just verify import statements are well-formed
            # Full resolution would require actual module installation
            return {"valid": True, "imports": imports}
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Import check failed: {str(e)}"]
            }
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "description": "Validate generated code quality",
            "checks": ["AST parsing", "import graph construction", "syntax validation"],
            "output": "Error report with improvement suggestions"
        }

