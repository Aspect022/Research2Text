"""
Pydantic schemas for the Research2Text pipeline.
Upgraded with confidence scoring for Phase 3 (Conformal Prediction).
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class TrainingConfig(BaseModel):
    optimizer: Optional[str] = Field(default=None)
    loss: Optional[str] = Field(default=None)
    epochs: Optional[int] = Field(default=None)
    learning_rate: Optional[float] = Field(default=None)
    batch_size: Optional[int] = Field(default=None)


class ConfidenceScore(BaseModel):
    """Confidence metadata for an extracted field.
    
    Implements a simplified conformal prediction framework:
    - `score` is the nonconformity score (0.0 = certain, 1.0 = uncertain)
    - `source` indicates where the value was extracted from
    - `evidence` stores the supporting text snippet from the paper
    """
    score: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Confidence score: 0.0 = low confidence (likely hallucinated), 1.0 = high confidence (explicitly stated)"
    )
    source: str = Field(
        default="heuristic",
        description="Extraction source: 'explicit' (directly stated), 'inferred' (LLM inferred), 'heuristic' (regex-based), 'default' (template value)"
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Supporting text snippet from the paper"
    )


class ArchitectureDetail(BaseModel):
    """Detailed architecture specification for code generation.

    Captures the structural information the code generator needs to
    produce architecture-specific (not generic) PyTorch code.
    """
    layer_types: List[str] = Field(
        default_factory=list,
        description="Ordered list of layer types, e.g. ['Conv2D', 'BiLSTM', 'Attention', 'Dense']",
    )
    input_shape: Optional[str] = Field(
        default=None,
        description="Input tensor shape, e.g. '(batch, 3, 224, 224)'",
    )
    output_shape: Optional[str] = Field(
        default=None,
        description="Output tensor shape, e.g. '(batch, num_classes)'",
    )
    num_classes: Optional[int] = Field(
        default=None,
        description="Number of output classes for classification tasks",
    )
    hidden_dims: List[int] = Field(
        default_factory=list,
        description="Hidden layer dimensions, e.g. [128, 256, 512]",
    )
    attention_type: Optional[str] = Field(
        default=None,
        description="Type of attention: 'self', 'cross', 'multi-head', 'additive', etc.",
    )
    preprocessing: List[str] = Field(
        default_factory=list,
        description="Data preprocessing steps, e.g. ['resize to 224x224', 'normalize']",
    )
    key_components: List[str] = Field(
        default_factory=list,
        description="Named components, e.g. ['ResBlock', 'BatchNorm', 'Dropout', 'SEBlock']",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_none_to_defaults(cls, data: Any) -> Any:
        """LLMs often return null for list fields — coerce to empty list."""
        if isinstance(data, dict):
            list_fields = ["layer_types", "hidden_dims", "preprocessing", "key_components"]
            for field in list_fields:
                if data.get(field) is None:
                    data[field] = []
        return data


class MethodStruct(BaseModel):
    algorithm_name: Optional[str] = Field(default=None)
    equations: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inputs: Optional[Dict[str, str]] = Field(default=None)
    outputs: Optional[Dict[str, str]] = Field(default=None)
    inputs_description: Optional[str] = Field(default=None)
    outputs_description: Optional[str] = Field(default=None)
    references: List[str] = Field(default_factory=list)

    # Architecture details for code generation
    architecture: ArchitectureDetail = Field(
        default_factory=ArchitectureDetail,
        description="Detailed architecture breakdown for generating accurate code",
    )
    paper_summary: Optional[str] = Field(
        default=None,
        description="2-3 sentence summary of the paper's proposed method",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_none_to_defaults(cls, data: Any) -> Any:
        # LLMs often return null for list/dict fields — coerce to empty list/dict.
        if isinstance(data, dict):
            list_fields = ["equations", "datasets", "references"]
            dict_fields = ["inputs", "outputs"]
            for field in list_fields:
                if data.get(field) is None:
                    data[field] = []
            for field in dict_fields:
                val = data.get(field)
                if val is None:
                    data[field] = {}
                elif isinstance(val, str):
                    # LLM returned a string instead of dict — move to _description field
                    data[f"{field}_description"] = val
                    data[field] = {}
        return data

    # Phase 3: Confidence scores for each extracted field
    confidence: Dict[str, ConfidenceScore] = Field(
        default_factory=dict,
        description="Per-field confidence scores from conformal prediction"
    )

    def overall_confidence(self) -> float:
        """Compute the mean confidence across all scored fields."""
        if not self.confidence:
            return 0.0
        scores = [c.score for c in self.confidence.values()]
        return sum(scores) / len(scores) if scores else 0.0

    def low_confidence_fields(self, threshold: float = 0.5) -> List[str]:
        """Return field names with confidence below the threshold."""
        return [
            field for field, conf in self.confidence.items()
            if conf.score < threshold
        ]


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
