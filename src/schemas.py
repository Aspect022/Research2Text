from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    optimizer: Optional[str] = Field(default=None)
    loss: Optional[str] = Field(default=None)
    epochs: Optional[int] = Field(default=None)
    learning_rate: Optional[float] = Field(default=None)
    batch_size: Optional[int] = Field(default=None)


class MethodStruct(BaseModel):
    algorithm_name: Optional[str] = Field(default=None)
    equations: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inputs: Dict[str, str] = Field(default_factory=dict)
    outputs: Dict[str, str] = Field(default_factory=dict)
    references: List[str] = Field(default_factory=list)


class GeneratedFile(BaseModel):
    path: str
    content: str


class RunResult(BaseModel):
    returncode: int
    stdout: str
    stderr: str


class ValidationResult(BaseModel):
    success: bool
    attempts: int
    last_error: Optional[str] = None
    logs_dir: Optional[str] = None

