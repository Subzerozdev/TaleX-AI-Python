"""Pydantic models for Kafka messages — schema matches Spring Boot DTOs."""

from pydantic import BaseModel, Field
from typing import Optional


class PipelineJobMessage(BaseModel):
    """Incoming job from Spring Boot."""
    model_config = {"populate_by_name": True}

    media_id: str = Field(alias="mediaId")
    s3_key: str = Field(alias="s3Key")
    s3_bucket: str = Field(alias="s3Bucket")
    media_type: str = Field(alias="mediaType")
    correlation_id: str = Field(alias="correlationId")
    requested_at: str = Field(alias="requestedAt")


class CopyrightViolationItem(BaseModel):
    """Single violation segment in copyright result."""
    model_config = {"populate_by_name": True}

    source_media_id: str = Field(alias="sourceMediaId")
    start_time_target: float = Field(alias="startTimeTarget")
    end_time_target: float = Field(alias="endTimeTarget")
    start_time_source: float = Field(alias="startTimeSource")
    end_time_source: float = Field(alias="endTimeSource")
    similarity_score: float = Field(alias="similarityScore")
    violation_type: str = Field(alias="violationType")


class CopyrightResultMessage(BaseModel):
    """Outgoing copyright result to Spring Boot."""
    model_config = {"populate_by_name": True}

    media_id: str = Field(alias="mediaId")
    correlation_id: str = Field(alias="correlationId")
    content_id: str = Field(alias="contentId")
    is_duplicate: bool = Field(alias="isDuplicate")
    overall_similarity: float = Field(alias="overallSimilarity")
    fingerprint_count: int = Field(alias="fingerprintCount")
    violations: list[CopyrightViolationItem] = []
    processed_at: str = Field(alias="processedAt")
    success: bool = True
    error_message: Optional[str] = Field(None, alias="errorMessage")


class ModerationViolationItem(BaseModel):
    """Single violation in moderation result."""
    model_config = {"populate_by_name": True}

    timestamp_ms: float = Field(alias="timestampMs")
    end_timestamp_ms: float = Field(alias="endTimestampMs")
    label: str
    confidence: float
    suggestion: str


class ModerationResultMessage(BaseModel):
    """Outgoing moderation result to Spring Boot."""
    model_config = {"populate_by_name": True}

    media_id: str = Field(alias="mediaId")
    correlation_id: str = Field(alias="correlationId")
    is_safe: bool = Field(alias="isSafe")
    primary_label: Optional[str] = Field(None, alias="primaryLabel")
    confidence_score: float = Field(alias="confidenceScore")
    violations: list[ModerationViolationItem] = []
    raw_response: str = Field(alias="rawResponse")
    processed_at: str = Field(alias="processedAt")
    success: bool = True
    error_message: Optional[str] = Field(None, alias="errorMessage")
