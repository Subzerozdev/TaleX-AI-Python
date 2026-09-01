"""Kafka topic constants — must match Spring Boot producer/consumer."""

from app.core.config import settings

_SUFFIX = settings.KAFKA_TOPIC_SUFFIX

TOPIC_PIPELINE_JOB = "content-pipeline-job" + _SUFFIX
TOPIC_COPYRIGHT_RESULT = "content-copyright-result" + _SUFFIX
TOPIC_MODERATION_JOB = "content-moderation-job" + _SUFFIX
TOPIC_MODERATION_RESULT = "content-moderation-result" + _SUFFIX
TOPIC_MEDIA_DELETE = "content-media-delete" + _SUFFIX

# Debezium CDC Topic
TOPIC_RECOMMENDATION_SYNC = "talex-cdc.public.series"
# Result Topic
TOPIC_RECOMMENDATION_RESULT = "recommendation-result"
