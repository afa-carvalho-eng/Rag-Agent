#!/usr/bin/env python3
"""CDK entry point. Run from the infra folder: cdk synth / cdk deploy."""

import aws_cdk as cdk

from rag_agent_stack import RagAgentStack

app = cdk.App()
RagAgentStack(app, "RagAgent")
app.synth()
