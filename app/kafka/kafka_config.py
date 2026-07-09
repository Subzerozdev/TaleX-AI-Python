"""Kafka topic constants — must match Spring Boot producer/consumer."""

TOPIC_PIPELINE_JOB = "content-pipeline-job"
TOPIC_COPYRIGHT_RESULT = "content-copyright-result"
TOPIC_MODERATION_JOB = "content-moderation-job"
TOPIC_MODERATION_RESULT = "content-moderation-result"

# Debezium CDC Topic
TOPIC_RECOMMENDATION_SYNC = "talex-cdc.public.series"
# Result Topic
TOPIC_RECOMMENDATION_RESULT = "recommendation-result"
