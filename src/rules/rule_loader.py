"""
Rule loader for agent knowledge bases.

Parses markdown rule files with embedded structured data
to provide comprehensive context to LLM agents.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """A pattern definition for extraction."""
    name: str
    regex: Optional[str]
    description: str
    confidence: float
    examples: List[Dict[str, str]]
    action: Optional[str] = None


@dataclass
class ValidationRule:
    """A validation rule for checking extracted data."""
    field: str
    rule_type: str  # "range", "regex", "custom"
    condition: str
    error_message: str


@dataclass
class EdgeCase:
    """An edge case with handling instructions."""
    name: str
    problem: str
    solution: str
    example: str


@dataclass
class AgentRules:
    """Complete ruleset for an agent."""
    agent_name: str
    version: str
    principles: List[str]
    patterns: List[Pattern]
    validation_rules: List[ValidationRule]
    edge_cases: List[EdgeCase]
    examples: List[Dict[str, Any]]
    raw_content: str  # Full markdown for LLM context


class RuleLoader:
    """
    Loads and parses rule files for agents.

    Rule files are markdown documents with structured sections:
    - ## Core Principles
    - ## Patterns
    - ## Validation Rules
    - ## Edge Cases
    - ## Examples
    """

    RULES_DIR = Path(__file__).parent.parent.parent / "data" / "rules"

    @classmethod
    def load(cls, agent_name: str) -> AgentRules:
        """
        Load rules for a specific agent.

        Args:
            agent_name: Name of the agent (e.g., "method_extraction")

        Returns:
            AgentRules object with parsed rules

        Raises:
            FileNotFoundError: If rules file doesn't exist
        """
        rule_file = cls.RULES_DIR / f"{agent_name}_rules.md"

        if not rule_file.exists():
            # Try without _rules suffix
            rule_file = cls.RULES_DIR / f"{agent_name}.md"

        if not rule_file.exists():
            raise FileNotFoundError(
                f"Rules not found for agent '{agent_name}'. "
                f"Searched: {rule_file}"
            )

        logger.info(f"Loading rules from {rule_file}")
        content = rule_file.read_text(encoding="utf-8")

        return cls._parse(content, agent_name)

    @classmethod
    def _parse(cls, content: str, agent_name: str) -> AgentRules:
        """Parse markdown content into structured rules."""
        # Extract version
        version = cls._extract_version(content) or "1.0.0"

        # Extract principles
        principles = cls._extract_principles(content)

        # Extract patterns
        patterns = cls._extract_patterns(content)

        # Extract validation rules
        validation_rules = cls._extract_validation_rules(content)

        # Extract edge cases
        edge_cases = cls._extract_edge_cases(content)

        # Extract examples
        examples = cls._extract_examples(content)

        return AgentRules(
            agent_name=agent_name,
            version=version,
            principles=principles,
            patterns=patterns,
            validation_rules=validation_rules,
            edge_cases=edge_cases,
            examples=examples,
            raw_content=content
        )

    @classmethod
    def _extract_version(cls, content: str) -> Optional[str]:
        """Extract version from rules file."""
        # Look for version patterns
        patterns = [
            r'Rules Version:\s*(\S+)',
            r'Version:\s*(\S+)',
            r'## Version\s*\n\s*(\S+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    @classmethod
    def _extract_principles(cls, content: str) -> List[str]:
        """Extract core principles from rules."""
        principles = []

        # Find Principles section
        section_match = re.search(
            r'##\s*Core Principles(.*?)(?=##|\Z)',
            content,
            re.DOTALL | re.IGNORECASE
        )

        if section_match:
            section = section_match.group(1)
            # Extract principle items
            principle_matches = re.findall(
                r'###\s*Principle \d+:\s*([^\n]+)',
                section
            )
            principles.extend(principle_matches)

            # Also look for bullet points
            bullet_matches = re.findall(
                r'^\s*[-*]\s*(.+)$',
                section,
                re.MULTILINE
            )
            principles.extend(bullet_matches)

        return principles

    @classmethod
    def _extract_patterns(cls, content: str) -> List[Pattern]:
        """Extract patterns from rules."""
        patterns = []

        # Find all Pattern sections
        pattern_sections = re.findall(
            r'###\s*Pattern \d+:\s*([^#]+?)(?=###|\Z)',
            content,
            re.DOTALL
        )

        for section in pattern_sections:
            pattern = cls._parse_pattern_section(section)
            if pattern:
                patterns.append(pattern)

        return patterns

    @classmethod
    def _parse_pattern_section(cls, section: str) -> Optional[Pattern]:
        """Parse a single pattern section."""
        # Extract name (first line)
        lines = section.strip().split('\n')
        if not lines:
            return None

        name = lines[0].strip()

        # Extract regex
        regex_match = re.search(r'\*\*Regex:\*\*\s*`([^`]+)`', section)
        regex = regex_match.group(1) if regex_match else None

        # Extract description
        desc_match = re.search(r'\*\*Description:\*\*\s*([^\n]+)', section)
        description = desc_match.group(1) if desc_match else ""

        # Extract confidence
        conf_match = re.search(r'\*\*Confidence:\*\*\s*(\d+\.?\d*)', section)
        confidence = float(conf_match.group(1)) if conf_match else 0.5

        # Extract examples
        examples = []
        example_matches = re.findall(
            r'```\s*\n?(.*?)\n?```',
            section,
            re.DOTALL
        )
        for ex in example_matches:
            examples.append({"code": ex.strip()})

        return Pattern(
            name=name,
            regex=regex,
            description=description,
            confidence=confidence,
            examples=examples
        )

    @classmethod
    def _extract_validation_rules(cls, content: str) -> List[ValidationRule]:
        """Extract validation rules."""
        rules = []

        # Find Validation Rules section
        section_match = re.search(
            r'##\s*Validation Rules(.*?)(?=##|\Z)',
            content,
            re.DOTALL | re.IGNORECASE
        )

        if section_match:
            section = section_match.group(1)

            # Extract rule definitions
            rule_matches = re.findall(
                r'\*\*Rule:\*\*\s*([^\n]+)',
                section
            )

            for match in rule_matches:
                # Parse rule
                parts = match.split(':')
                if len(parts) >= 2:
                    rules.append(ValidationRule(
                        field=parts[0].strip(),
                        rule_type="custom",
                        condition=match,
                        error_message=f"Validation failed: {match}"
                    ))

        return rules

    @classmethod
    def _extract_edge_cases(cls, content: str) -> List[EdgeCase]:
        """Extract edge cases."""
        edge_cases = []

        # Find Edge Cases section
        section_match = re.search(
            r'##\s*Edge Cases(.*?)(?=##|\Z)',
            content,
            re.DOTALL | re.IGNORECASE
        )

        if section_match:
            section = section_match.group(1)

            # Extract edge case sections
            case_matches = re.findall(
                r'###\s*Edge Case \d+:\s*([^#]+)',
                section,
                re.DOTALL
            )

            for case_content in case_matches:
                lines = case_content.strip().split('\n')
                if not lines:
                    continue

                name = lines[0].strip()

                # Extract problem
                prob_match = re.search(
                    r'\*\*Problem:\*\*\s*([^\n]+)',
                    case_content
                )
                problem = prob_match.group(1) if prob_match else ""

                # Extract solution
                sol_match = re.search(
                    r'\*\*Solution:\*\*\s*([^\n]+)',
                    case_content
                )
                solution = sol_match.group(1) if sol_match else ""

                # Extract example
                ex_match = re.search(
                    r'```\s*\n?(.*?)\n?```',
                    case_content,
                    re.DOTALL
                )
                example = ex_match.group(1) if ex_match else ""

                edge_cases.append(EdgeCase(
                    name=name,
                    problem=problem,
                    solution=solution,
                    example=example
                ))

        return edge_cases

    @classmethod
    def _extract_examples(cls, content: str) -> List[Dict[str, Any]]:
        """Extract examples from rules."""
        examples = []

        # Find Examples section
        section_match = re.search(
            r'##\s*Examples(.*?)(?=##|\Z)',
            content,
            re.DOTALL | re.IGNORECASE
        )

        if section_match:
            section = section_match.group(1)

            # Extract example sections
            example_matches = re.findall(
                r'###\s*Example \d+:\s*([^#]+)',
                section,
                re.DOTALL
            )

            for ex_content in example_matches:
                example = {}

                # Extract input
                input_match = re.search(
                    r'\*\*Input:\*\*\s*```\s*\n?(.*?)\n?```',
                    ex_content,
                    re.DOTALL
                )
                if input_match:
                    example["input"] = input_match.group(1).strip()

                # Extract output
                output_match = re.search(
                    r'\*\*Output:\*\*\s*```json\s*\n?(.*?)\n?```',
                    ex_content,
                    re.DOTALL
                )
                if output_match:
                    try:
                        example["output"] = json.loads(output_match.group(1))
                    except json.JSONDecodeError:
                        example["output"] = output_match.group(1).strip()

                if example:
                    examples.append(example)

        return examples

    @classmethod
    def format_for_llm(cls, rules: AgentRules, max_chars: int = 8000) -> str:
        """
        Format rules for LLM context window.

        Args:
            rules: AgentRules object
            max_chars: Maximum characters to include

        Returns:
            Formatted string for LLM prompt
        """
        lines = [
            f"# {rules.agent_name.upper()} RULES",
            f"Version: {rules.version}",
            "",
            "## Core Principles",
        ]

        for principle in rules.principles[:10]:  # Limit principles
            lines.append(f"- {principle}")

        lines.extend(["", "## Key Patterns"])

        for pattern in rules.patterns[:20]:  # Limit patterns
            lines.append(f"\n### {pattern.name}")
            if pattern.regex:
                lines.append(f"Pattern: `{pattern.regex}`")
            lines.append(f"Confidence: {pattern.confidence}")
            if pattern.description:
                lines.append(f"Description: {pattern.description}")

        lines.extend(["", "## Edge Cases to Handle"])

        for edge_case in rules.edge_cases[:10]:
            lines.append(f"\n- {edge_case.name}: {edge_case.problem}")
            lines.append(f"  Solution: {edge_case.solution}")

        result = "\n".join(lines)

        # Truncate if needed
        if len(result) > max_chars:
            result = result[:max_chars] + "\n\n[Rules truncated...]"

        return result


# Convenience function
def load_rules_for_agent(agent_name: str) -> AgentRules:
    """Load rules for a specific agent."""
    return RuleLoader.load(agent_name)
