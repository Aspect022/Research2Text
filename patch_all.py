import os

path_schemas = "D:/Projects/Research2Text-main/src/schemas.py"
with open(path_schemas, "r", encoding="utf-8") as f:
    code = f.read()

target_schemas = """    # Phase 3: Confidence scores for each extracted field
    confidence: Dict[str, ConfidenceScore] = Field(
        default_factory=dict,
        description="Per-field confidence scores from conformal prediction"
    )"""

replacement_schemas = """    @model_validator(mode="before")
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
                if data.get(field) is None:
                    data[field] = {}
        return data

    # Phase 3: Confidence scores for each extracted field
    confidence: Dict[str, ConfidenceScore] = Field(
        default_factory=dict,
        description="Per-field confidence scores from conformal prediction"
    )"""

if target_schemas in code:
    code = code.replace(target_schemas, replacement_schemas)
    with open(path_schemas, "w", encoding="utf-8") as f:
        f.write(code)
    print("Successfully patched schemas.py")
else:
    print("Failed to patch schemas.py")


path_codegen = "D:/Projects/Research2Text-main/src/code_generator.py"
with open(path_codegen, "r", encoding="utf-8") as f:
    code = f.read()

target_codegen_1 = """9. Do NOT leave TODO placeholders or incomplete code."""

replacement_codegen_1 = """9. Do NOT leave TODO placeholders or incomplete code.
10. Use ONLY ASCII characters in code and print statements (e.g., use '->' instead of '→') to prevent UnicodeEncodeError in Windows."""

if target_codegen_1 in code:
    code = code.replace(target_codegen_1, replacement_codegen_1)
    with open(path_codegen, "w", encoding="utf-8") as f:
        f.write(code)
    print("Successfully patched system prompt in code_generator.py")
else:
    print("Failed to patch code_generator.py prompt")
