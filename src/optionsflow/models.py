from typing import Annotated, Any, Literal, TypedDict

from pydantic import Field

Symbol = Annotated[
    str,
    Field(
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9.^=-]+$",
        description="Yahoo Finance ticker symbol",
    ),
]
Strategy = Annotated[
    Literal["ccs", "pcs", "csp", "cc"],
    Field(
        description="Options strategy: call credit spread, put credit spread, cash-secured put, or covered call"
    ),
]
Expiration = Annotated[
    str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Expiration date in YYYY-MM-DD format")
]
DeltaTarget = Annotated[
    float, Field(gt=0, lt=1, description="Absolute target delta for CSP/CC selection")
]
WidthPct = Annotated[
    float, Field(gt=0, le=0.5, description="Target spread width as a fraction of the short strike")
]


class ProviderMetadata(TypedDict):
    name: str
    as_of: str
    real_time: bool


class ResponseEnvelope(TypedDict):
    success: bool
    timestamp: str
    data: dict[str, Any]
    provider: ProviderMetadata
    warnings: list[str]
