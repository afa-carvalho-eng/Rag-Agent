"""AWS infrastructure for the RAG agent, as one CDK stack.

Ingestion:  S3 -> SQS -> Lambda -> Bedrock embeddings -> OpenSearch Serverless
Questions:  API Gateway (Cognito JWT) -> VPC link -> ALB -> ECS Fargate
            -> DynamoDB (history), OpenSearch (retrieval), Bedrock (Claude)
"""

from __future__ import annotations

import json
from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_apigatewayv2 as apigwv2
from aws_cdk import aws_apigatewayv2_authorizers as authorizers
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_ecs_patterns as ecs_patterns
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_lambda_event_sources as event_sources
from aws_cdk import aws_logs as logs
from aws_cdk import aws_opensearchserverless as aoss
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_notifications as s3n
from aws_cdk import aws_sqs as sqs
from constructs import Construct

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
COLLECTION_NAME = "rag-agent"
INDEX_NAME = "documents"
INGEST_TIMEOUT = Duration.minutes(5)


class RagAgentStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- Storage -------------------------------------------------------
        documents = s3.Bucket(
            self,
            "Documents",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        history = dynamodb.Table(
            self,
            "History",
            partition_key=dynamodb.Attribute(name="session_id", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="ts", type=dynamodb.AttributeType.NUMBER),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="expires_at",
            removal_policy=RemovalPolicy.DESTROY,
        )

        # --- Ingestion queue -----------------------------------------------
        dead_letters = sqs.Queue(
            self, "IngestDeadLetters", retention_period=Duration.days(14), enforce_ssl=True
        )
        ingest_queue = sqs.Queue(
            self,
            "IngestQueue",
            # AWS recommends at least six times the consumer's timeout.
            visibility_timeout=Duration.seconds(INGEST_TIMEOUT.to_seconds() * 6),
            dead_letter_queue=sqs.DeadLetterQueue(max_receive_count=3, queue=dead_letters),
            enforce_ssl=True,
        )
        documents.add_event_notification(
            s3.EventType.OBJECT_CREATED, s3n.SqsDestination(ingest_queue)
        )

        # --- Roles (created first: the vector index policy names them) ------
        ingest_role = iam.Role(
            self,
            "IngestRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )
        agent_role = iam.Role(
            self, "AgentTaskRole", assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
        )

        # --- Vector index: OpenSearch Serverless ----------------------------
        collection_resource = f"collection/{COLLECTION_NAME}"
        encryption_policy = aoss.CfnSecurityPolicy(
            self,
            "VectorEncryption",
            name=f"{COLLECTION_NAME}-enc",
            type="encryption",
            policy=json.dumps(
                {
                    "Rules": [{"ResourceType": "collection", "Resource": [collection_resource]}],
                    "AWSOwnedKey": True,
                }
            ),
        )
        # The endpoint is reachable from the internet but every request must be
        # signed by a principal listed in the data access policy below. A VPC
        # endpoint would remove the public route; see the README.
        network_policy = aoss.CfnSecurityPolicy(
            self,
            "VectorNetwork",
            name=f"{COLLECTION_NAME}-net",
            type="network",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {"ResourceType": "collection", "Resource": [collection_resource]}
                        ],
                        "AllowFromPublic": True,
                    }
                ]
            ),
        )
        collection = aoss.CfnCollection(self, "Vectors", name=COLLECTION_NAME, type="VECTORSEARCH")
        # A collection cannot be created before its encryption policy exists.
        collection.node.add_dependency(encryption_policy)
        collection.node.add_dependency(network_policy)

        aoss.CfnAccessPolicy(
            self,
            "VectorAccess",
            name=f"{COLLECTION_NAME}-data",
            type="data",
            policy=self.to_json_string(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "index",
                                "Resource": [f"index/{COLLECTION_NAME}/*"],
                                "Permission": [
                                    "aoss:CreateIndex",
                                    "aoss:DescribeIndex",
                                    "aoss:ReadDocument",
                                    "aoss:WriteDocument",
                                ],
                            }
                        ],
                        "Principal": [ingest_role.role_arn, agent_role.role_arn],
                    }
                ]
            ),
        )

        vector_access = iam.PolicyStatement(
            actions=["aoss:APIAccessAll"], resources=[collection.attr_arn]
        )
        # NOTE: confirm the IAM actions for the Bedrock Messages endpoint used by
        # AnthropicBedrockMantle against the AWS documentation before deploying,
        # and narrow `resources` to the model ARNs in use.
        model_access = iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            resources=["*"],
        )
        for role in (ingest_role, agent_role):
            role.add_to_policy(vector_access)
            role.add_to_policy(model_access)

        shared_environment = {
            "RAG_BACKEND": "aws",
            "OPENSEARCH_ENDPOINT": collection.attr_collection_endpoint,
            "OPENSEARCH_INDEX": INDEX_NAME,
            "HISTORY_TABLE": history.table_name,
        }

        # --- Ingestion function (same container image as the agent) ---------
        ingest = lambda_.DockerImageFunction(
            self,
            "Ingest",
            code=lambda_.DockerImageCode.from_image_asset(
                REPO_ROOT,
                entrypoint=["python", "-m", "awslambdaric"],
                cmd=["ragagent.ingest.handler"],
            ),
            role=ingest_role,
            timeout=INGEST_TIMEOUT,
            memory_size=1024,
            environment=shared_environment,
            log_group=logs.LogGroup(self, "IngestLogs", retention=logs.RetentionDays.ONE_MONTH),
        )
        ingest.add_event_source(
            event_sources.SqsEventSource(
                ingest_queue, batch_size=5, report_batch_item_failures=True
            )
        )
        documents.grant_read(ingest_role)

        # --- Agent service on Fargate, behind an internal load balancer -----
        vpc = ec2.Vpc(self, "Vpc", max_azs=2, nat_gateways=1)
        cluster = ecs.Cluster(self, "Cluster", vpc=vpc)
        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "Agent",
            cluster=cluster,
            public_load_balancer=False,
            cpu=512,
            memory_limit_mib=1024,
            desired_count=1,
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_asset(REPO_ROOT),
                container_port=8000,
                task_role=agent_role,
                environment=shared_environment,
                log_driver=ecs.LogDrivers.aws_logs(
                    stream_prefix="agent", log_retention=logs.RetentionDays.ONE_MONTH
                ),
            ),
        )
        service.target_group.configure_health_check(path="/health")
        history.grant_read_write_data(agent_role)
        documents.grant_read(agent_role)

        # --- Public entry point: HTTP API with Cognito authentication -------
        users = cognito.UserPool(
            self,
            "Users",
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True),
            removal_policy=RemovalPolicy.DESTROY,
        )
        app_client = users.add_client("AppClient", auth_flows=cognito.AuthFlow(user_srp=True))

        link_security_group = ec2.SecurityGroup(
            self, "VpcLinkSecurityGroup", vpc=vpc, allow_all_outbound=True
        )
        service.load_balancer.connections.allow_from(
            link_security_group, ec2.Port.tcp(80), "API Gateway VPC link"
        )
        vpc_link = apigwv2.VpcLink(self, "VpcLink", vpc=vpc, security_groups=[link_security_group])

        api = apigwv2.HttpApi(self, "Api")
        api.add_routes(
            path="/chat",
            methods=[apigwv2.HttpMethod.POST],
            integration=integrations.HttpAlbIntegration(
                "AgentIntegration", service.listener, vpc_link=vpc_link
            ),
            authorizer=authorizers.HttpUserPoolAuthorizer(
                "CognitoAuthorizer", users, user_pool_clients=[app_client]
            ),
        )

        CfnOutput(self, "ApiUrl", value=api.api_endpoint)
        CfnOutput(self, "DocumentsBucket", value=documents.bucket_name)
        CfnOutput(self, "UserPoolId", value=users.user_pool_id)
        CfnOutput(self, "UserPoolClientId", value=app_client.user_pool_client_id)
        CfnOutput(self, "IngestDeadLetterQueue", value=dead_letters.queue_name)
