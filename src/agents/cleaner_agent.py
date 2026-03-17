"""
Agent 10: Cleaner Agent
Responsibility: Clean up outdated chunks, refresh RAG index, and maintain database hygiene
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import sys
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
from .base import BaseAgent, AgentMessage, AgentResponse


class CleanerAgent(BaseAgent):
    """Agent for cleaning outdated chunks and maintaining RAG database."""

    def __init__(self):
        super().__init__("cleaner", "Cleaner Agent")
        self.project_root = Path(__file__).resolve().parents[2]
        self.raw_text_dir = self.project_root / "data" / "raw_texts"
        self.chroma_dir = self.project_root / "data" / "chroma_db"
        self.collection_name = "research_papers"

    def _get_collection_name(self) -> str:
        """Get collection name based on current embedder dimension."""
        try:
            sys.path.insert(0, str(self.project_root / "src"))
            from embeddings import get_embedder
            embedder = get_embedder(prefer_gemini=True)
            test_embedding = embedder.embed(["test"])
            dim = len(test_embedding[0])
            return f"{self.collection_name}_{dim}"
        except Exception:
            return self.collection_name
    
    def process(self, message: AgentMessage) -> AgentResponse:
        """Clean outdated chunks and refresh the index."""
        payload = message.payload
        action = payload.get("action", "clean_old")  # clean_old, clean_base, refresh_index, full_clean
        
        try:
            if action == "clean_old":
                result = self._clean_old_chunks(
                    days_old=payload.get("days_old", 30),
                    dry_run=payload.get("dry_run", False)
                )
            elif action == "clean_base":
                result = self._clean_base_chunks(
                    base_name=payload.get("base_name"),
                    dry_run=payload.get("dry_run", False)
                )
            elif action == "refresh_index":
                result = self._refresh_index()
            elif action == "full_clean":
                result = self._full_clean(
                    days_old=payload.get("days_old", 30),
                    dry_run=payload.get("dry_run", False)
                )
            else:
                return AgentResponse(
                    success=False,
                    error=f"Unknown action: {action}. Use: clean_old, clean_base, refresh_index, full_clean"
                )
            
            return AgentResponse(
                success=True,
                data=result
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Cleaning failed: {str(e)}"
            )
    
    def _clean_old_chunks(self, days_old: int = 30, dry_run: bool = False) -> Dict[str, Any]:
        """Remove chunks older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_files = []
        deleted_ids = []
        
        # Get all chunk files
        chunk_files = list(self.raw_text_dir.glob("*_chunk_*.txt"))
        
        for chunk_file in chunk_files:
            try:
                # Check file modification time
                mtime = datetime.fromtimestamp(chunk_file.stat().st_mtime)
                if mtime < cutoff_date:
                    # Extract base and chunk_id
                    base_name = chunk_file.stem.split("_chunk_")[0]
                    chunk_id = int(chunk_file.stem.split("_chunk_")[-1])
                    chunk_id_str = f"{base_name}:{chunk_id}"
                    
                    deleted_files.append(str(chunk_file))
                    deleted_ids.append(chunk_id_str)
                    
                    if not dry_run:
                        chunk_file.unlink()
            except (ValueError, OSError) as e:
                continue
        
        # Remove from ChromaDB
        if deleted_ids and not dry_run:
            self._remove_from_chromadb(deleted_ids)
        
        return {
            "action": "clean_old",
            "days_old": days_old,
            "dry_run": dry_run,
            "deleted_files": len(deleted_files),
            "deleted_ids": len(deleted_ids),
            "files": deleted_files[:10] if dry_run else [],  # Show first 10 in dry run
        }
    
    def _clean_base_chunks(self, base_name: str, dry_run: bool = False) -> Dict[str, Any]:
        """Remove all chunks for a specific base."""
        if not base_name:
            return {"error": "base_name is required"}
        
        deleted_files = []
        deleted_ids = []
        
        # Find all chunk files for this base
        chunk_files = list(self.raw_text_dir.glob(f"{base_name}_chunk_*.txt"))
        
        for chunk_file in chunk_files:
            try:
                chunk_id = int(chunk_file.stem.split("_chunk_")[-1])
                chunk_id_str = f"{base_name}:{chunk_id}"
                
                deleted_files.append(str(chunk_file))
                deleted_ids.append(chunk_id_str)
                
                if not dry_run:
                    chunk_file.unlink()
            except (ValueError, OSError):
                continue
        
        # Also delete the main text file if it exists
        main_file = self.raw_text_dir / f"{base_name}.txt"
        if main_file.exists():
            deleted_files.append(str(main_file))
            if not dry_run:
                main_file.unlink()
        
        # Remove from ChromaDB
        if deleted_ids and not dry_run:
            self._remove_from_chromadb(deleted_ids)
        
        return {
            "action": "clean_base",
            "base_name": base_name,
            "dry_run": dry_run,
            "deleted_files": len(deleted_files),
            "deleted_ids": len(deleted_ids),
            "files": deleted_files if dry_run else [],
        }
    
    def _refresh_index(self) -> Dict[str, Any]:
        """Refresh the ChromaDB index by re-indexing all current chunks."""
        try:
            # Import index function
            sys.path.insert(0, str(self.project_root / "src"))
            from index_documents import index_documents
            
            # Re-index everything
            index_documents()
            
            # Get current stats
            stats = self._get_index_stats()
            
            return {
                "action": "refresh_index",
                "success": True,
                "stats": stats
            }
        except Exception as e:
            return {
                "action": "refresh_index",
                "success": False,
                "error": str(e)
            }
    
    def _full_clean(self, days_old: int = 30, dry_run: bool = False) -> Dict[str, Any]:
        """Perform full cleanup: remove old chunks, orphaned files, and refresh index."""
        results = {
            "action": "full_clean",
            "dry_run": dry_run,
            "steps": {}
        }
        
        # Step 1: Clean old chunks
        old_result = self._clean_old_chunks(days_old=days_old, dry_run=dry_run)
        results["steps"]["clean_old"] = old_result
        
        # Step 2: Remove orphaned chunks (chunks in DB but no file, or vice versa)
        orphan_result = self._remove_orphaned_chunks(dry_run=dry_run)
        results["steps"]["remove_orphans"] = orphan_result
        
        # Step 3: Refresh index
        if not dry_run:
            refresh_result = self._refresh_index()
            results["steps"]["refresh_index"] = refresh_result
        
        return results
    
    def _remove_orphaned_chunks(self, dry_run: bool = False) -> Dict[str, Any]:
        """Remove chunks that exist in DB but not in files, or vice versa."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            try:
                collection = client.get_collection(
                    name=self.collection_name,
                    embedding_function=embed_fn
                )
            except Exception:
                return {"orphaned_found": 0, "message": "Collection does not exist"}
            
            # Get all IDs from ChromaDB
            all_results = collection.get()
            db_ids = set(all_results.get("ids", []))
            
            # Get all chunk files
            chunk_files = list(self.raw_text_dir.glob("*_chunk_*.txt"))
            file_ids = set()
            
            for chunk_file in chunk_files:
                try:
                    base_name = chunk_file.stem.split("_chunk_")[0]
                    chunk_id = int(chunk_file.stem.split("_chunk_")[-1])
                    file_ids.add(f"{base_name}:{chunk_id}")
                except (ValueError, IndexError):
                    continue
            
            # Find orphaned: in DB but not in files
            orphaned_db_ids = db_ids - file_ids
            
            # Find orphaned: in files but not in DB (less common, but possible)
            orphaned_file_ids = file_ids - db_ids
            
            removed_count = 0
            if orphaned_db_ids and not dry_run:
                collection.delete(ids=list(orphaned_db_ids))
                removed_count = len(orphaned_db_ids)
            
            return {
                "orphaned_in_db": len(orphaned_db_ids),
                "orphaned_in_files": len(orphaned_file_ids),
                "removed": removed_count,
                "dry_run": dry_run
            }
        except Exception as e:
            return {
                "error": str(e),
                "orphaned_found": 0
            }
    
    def _remove_from_chromadb(self, ids: List[str]) -> None:
        """Remove specific IDs from ChromaDB."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            try:
                collection = client.get_collection(
                    name=self.collection_name,
                    embedding_function=embed_fn
                )
                collection.delete(ids=ids)
            except Exception:
                # Collection might not exist, which is fine
                pass
        except Exception:
            # ChromaDB might not be available, fail silently
            pass
    
    def _get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            try:
                collection = client.get_collection(
                    name=self.collection_name,
                    embedding_function=embed_fn
                )
                count = collection.count()
                
                # Get unique bases
                all_results = collection.get()
                bases = set()
                for metadata in all_results.get("metadatas", []):
                    if metadata and "base" in metadata:
                        bases.add(metadata["base"])
                
                return {
                    "total_chunks": count,
                    "unique_bases": len(bases),
                    "bases": sorted(list(bases))
                }
            except Exception:
                return {"total_chunks": 0, "unique_bases": 0, "bases": []}
        except Exception:
            return {"total_chunks": 0, "unique_bases": 0, "bases": []}
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "description": "Clean outdated chunks, refresh RAG index, and maintain database hygiene",
            "actions": {
                "clean_old": "Remove chunks older than N days",
                "clean_base": "Remove all chunks for a specific paper base",
                "refresh_index": "Re-index all current chunks in ChromaDB",
                "full_clean": "Complete cleanup: old chunks, orphans, and refresh"
            },
            "features": [
                "Age-based cleanup",
                "Base-specific cleanup",
                "Orphan detection and removal",
                "Index refresh",
                "Dry-run mode for safety"
            ]
        }

