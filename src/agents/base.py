"""
Base classes for multi-agent system.
Defines the interface and message schemas for all agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """Standard message format for inter-agent communication."""
    agent_id: str = Field(description="ID of the sending agent")
    message_type: str = Field(description="Type of message: request, response, error")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    correlation_id: Optional[str] = Field(default=None, description="For tracking request-response pairs")


class AgentResponse(BaseModel):
    """Standard response format from agents."""
    success: bool = Field(description="Whether the operation succeeded")
    data: Dict[str, Any] = Field(default_factory=dict, description="Response data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    processing_time: Optional[float] = Field(default=None, description="Time taken in seconds")


class BaseAgent(ABC):
    """
    Base class for all agents in the Research2Text system.
    Each agent implements a specific capability with a standardized interface.
    """
    
    def __init__(self, agent_id: str, agent_name: str):
        self.agent_id = agent_id
        self.agent_name = agent_name
    
    @abstractmethod
    def process(self, message: AgentMessage) -> AgentResponse:
        """
        Process a message and return a response.
        This is the main entry point for agent operations.
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return a description of what this agent can do.
        Used by the orchestrator for task routing.
        """
        pass
    
    def validate_message(self, message: AgentMessage) -> bool:
        """Validate that a message is appropriate for this agent."""
        return True
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, name={self.agent_name})"

