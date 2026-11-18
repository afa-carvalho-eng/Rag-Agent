# Deployment guide

This is a fictional handbook written as sample data for the RAG agent.

All services deploy through the continuous delivery pipeline. A merge to the main branch builds a container image, runs the unit and integration test suites, and deploys the image to the staging environment automatically.

Promotion from staging to production requires a manual approval from a second engineer who did not write the change. Production deployments are frozen on Fridays after 14:00 and during the last week of December.

Releases roll out as a canary: five percent of traffic for 20 minutes, then the remaining traffic if the error rate stays below one percent. The pipeline rolls back on its own when the canary breaches that threshold.

Database schema migrations ship separately from application code and must be backward compatible with the previous release, so a rollback never needs a schema change.
