from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalyzeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9.^=-]+$")
    expiration_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    width_pct: float = Field(default=0.05, gt=0, le=0.5)

    @field_validator("symbol")
    @classmethod
    def normalize(cls, v):
        return v.strip().upper()
