"""
Convenience script for cleaning RAG database.
Can be used from command line or imported as a module.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from agents.cleaner_agent import CleanerAgent
from agents.base import AgentMessage


def clean_old_chunks(days_old: int = 30, dry_run: bool = False):
    """Remove chunks older than specified days."""
    cleaner = CleanerAgent()
    msg = AgentMessage(
        agent_id="user",
        message_type="request",
        payload={"action": "clean_old", "days_old": days_old, "dry_run": dry_run}
    )
    return cleaner.process(msg)


def clean_base(base_name: str, dry_run: bool = False):
    """Remove all chunks for a specific base."""
    cleaner = CleanerAgent()
    msg = AgentMessage(
        agent_id="user",
        message_type="request",
        payload={"action": "clean_base", "base_name": base_name, "dry_run": dry_run}
    )
    return cleaner.process(msg)


def refresh_index():
    """Refresh the ChromaDB index."""
    cleaner = CleanerAgent()
    msg = AgentMessage(
        agent_id="user",
        message_type="request",
        payload={"action": "refresh_index"}
    )
    return cleaner.process(msg)


def full_clean(days_old: int = 30, dry_run: bool = False):
    """Perform full cleanup."""
    cleaner = CleanerAgent()
    msg = AgentMessage(
        agent_id="user",
        message_type="request",
        payload={"action": "full_clean", "days_old": days_old, "dry_run": dry_run}
    )
    return cleaner.process(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean RAG database and remove outdated chunks")
    parser.add_argument("action", choices=["clean_old", "clean_base", "refresh", "full_clean"],
                       help="Action to perform")
    parser.add_argument("--days", type=int, default=30, help="Days old for clean_old action")
    parser.add_argument("--base", type=str, help="Base name for clean_base action")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--apply", action="store_true", help="Actually apply changes (opposite of dry-run)")
    
    args = parser.parse_args()
    dry_run = args.dry_run and not args.apply
    
    if args.action == "clean_old":
        result = clean_old_chunks(days_old=args.days, dry_run=dry_run)
    elif args.action == "clean_base":
        if not args.base:
            print("Error: --base is required for clean_base action")
            sys.exit(1)
        result = clean_base(base_name=args.base, dry_run=dry_run)
    elif args.action == "refresh":
        result = refresh_index()
    elif args.action == "full_clean":
        result = full_clean(days_old=args.days, dry_run=dry_run)
    
    if result.success:
        print("✅ Success!")
        import json
        print(json.dumps(result.data, indent=2))
    else:
        print(f"❌ Failed: {result.error}")
        sys.exit(1)

