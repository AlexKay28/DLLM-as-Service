"""
Schema definition for FlowFill evaluation.

12 typed slots used across all four LOGHUB corpora.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class FieldType(Enum):
    STRING = "string"
    FLOAT = "float"
    BOOL = "bool"
    DATETIME = "datetime"
    ENUM = "enum"


class FieldCategory(Enum):
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    ANOMALY = "anomaly"


@dataclass
class SchemaField:
    name: str
    type: FieldType
    category: FieldCategory
    required: bool = True
    enum_values: Optional[list[str]] = None


SCHEMA_FIELDS: list[SchemaField] = [
    SchemaField("timestamp_norm", FieldType.DATETIME, FieldCategory.STRUCTURAL),
    SchemaField("host", FieldType.STRING, FieldCategory.STRUCTURAL),
    SchemaField("component", FieldType.STRING, FieldCategory.STRUCTURAL),
    SchemaField("severity", FieldType.ENUM, FieldCategory.STRUCTURAL,
                enum_values=["DEBUG", "INFO", "WARN", "ERROR", "FATAL", "CRITICAL"]),
    SchemaField("event_type", FieldType.STRING, FieldCategory.SEMANTIC),
    SchemaField("message_template", FieldType.STRING, FieldCategory.STRUCTURAL),
    SchemaField("error_code", FieldType.STRING, FieldCategory.STRUCTURAL),
    SchemaField("affected_entity", FieldType.STRING, FieldCategory.SEMANTIC),
    SchemaField("metric_value", FieldType.FLOAT, FieldCategory.STRUCTURAL),
    SchemaField("is_anomaly", FieldType.BOOL, FieldCategory.ANOMALY),
    SchemaField("anomaly_score", FieldType.FLOAT, FieldCategory.ANOMALY),
    SchemaField("remediation_hint", FieldType.STRING, FieldCategory.SEMANTIC),
]

FIELD_NAMES = [f.name for f in SCHEMA_FIELDS]
FIELD_BY_NAME = {f.name: f for f in SCHEMA_FIELDS}


@dataclass
class Record:
    """A single structured log record."""
    fields: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str, default: Any = None) -> Any:
        return self.fields.get(name, default)

    def set(self, name: str, value: Any) -> None:
        self.fields[name] = value

    def has(self, name: str) -> bool:
        return name in self.fields and self.fields[name] is not None
