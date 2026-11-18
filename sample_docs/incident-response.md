# Incident response

This is a fictional handbook written as sample data for the RAG agent.

An incident is any unplanned interruption or degradation of a production service. Incidents are classified by severity. Severity 1 means a complete outage or data loss affecting customers. Severity 2 means a major feature is unavailable. Severity 3 covers minor degradation with a workaround.

The on-call engineer must acknowledge a severity 1 page within 15 minutes and open an incident channel named after the incident number. An incident commander is appointed for every severity 1 and severity 2 incident and is the only person who decides on rollbacks.

Customers are notified through the status page within 30 minutes of a confirmed severity 1 incident, and updates follow every hour until resolution.

A written postmortem is due within five working days of resolution. Postmortems are blameless: they describe what happened and which safeguards failed, not who made a mistake.
