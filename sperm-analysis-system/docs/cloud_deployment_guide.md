# ☁️ Cloud Deployment Guide

> **Developer:** Youssef Shitiwi  
> **System:** Sperm Analysis with AI-Powered CASA Metrics

This comprehensive guide covers deploying the sperm analysis system to major cloud platforms with production-ready configurations, scalability, and monitoring.

## 📋 Table of Contents

1. [AWS Deployment](#aws-deployment)
2. [Google Cloud Platform](#google-cloud-platform)
3. [Microsoft Azure](#microsoft-azure)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Serverless Options](#serverless-options)
6. [Multi-Cloud Strategy](#multi-cloud-strategy)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring & Observability](#monitoring--observability)
9. [Security & Compliance](#security--compliance)
10. [Cost Optimization](#cost-optimization)

---

## 🌟 AWS Deployment

### ECS with Fargate Setup

Create `deploy/aws/ecs-task-definition.json`:

```json
{
  "family": "sperm-analysis-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT_ID:role/spermAnalysisTaskRole",
  "containerDefinitions": [
    {
      "name": "sperm-analysis-api",
      "image": "YOUR_ECR_REPO/sperm-analysis:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://username:password@rds-endpoint:5432/sperm_analysis"
        },
        {
          "name": "REDIS_URL",
          "value": "redis://elasticache-endpoint:6379/0"
        },
        {
          "name": "AWS_DEFAULT_REGION",
          "value": "us-west-2"
        }
      ],
      "secrets": [
        {
          "name": "SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-west-2:ACCOUNT_ID:secret:sperm-analysis-secrets:SecretKey::"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/sperm-analysis",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "mountPoints": [
        {
          "sourceVolume": "efs-storage",
          "containerPath": "/app/data",
          "readOnly": false
        }
      ]
    }
  ],
  "volumes": [
    {
      "name": "efs-storage",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-XXXXXXXX",
        "transitEncryption": "ENABLED",
        "authorizationConfig": {
          "accessPointId": "fsap-XXXXXXXX"
        }
      }
    }
  ]
}
```

### CloudFormation Template

Create `deploy/aws/cloudformation.yaml`:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Sperm Analysis System - AWS Infrastructure by Youssef Shitiwi'

Parameters:
  Environment:
    Type: String
    Default: production
    AllowedValues: [development, staging, production]
  
  VpcCIDR:
    Type: String
    Default: 10.0.0.0/16
  
  DatabasePassword:
    Type: String
    NoEcho: true
    MinLength: 8

Resources:
  # VPC and Networking
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: !Ref VpcCIDR
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub ${Environment}-sperm-analysis-vpc
        - Key: Developer
          Value: Youssef Shitiwi

  PublicSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs '']
      CidrBlock: 10.0.1.0/24
      MapPublicIpOnLaunch: true

  PublicSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs '']
      CidrBlock: 10.0.2.0/24
      MapPublicIpOnLaunch: true

  PrivateSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs '']
      CidrBlock: 10.0.10.0/24

  PrivateSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs '']
      CidrBlock: 10.0.20.0/24

  # Internet Gateway
  InternetGateway:
    Type: AWS::EC2::InternetGateway

  InternetGatewayAttachment:
    Type: AWS::EC2::VPCGatewayAttachment
    Properties:
      InternetGatewayId: !Ref InternetGateway
      VpcId: !Ref VPC

  # NAT Gateway
  NatGateway1EIP:
    Type: AWS::EC2::EIP
    DependsOn: InternetGatewayAttachment
    Properties:
      Domain: vpc

  NatGateway1:
    Type: AWS::EC2::NatGateway
    Properties:
      AllocationId: !GetAtt NatGateway1EIP.AllocationId
      SubnetId: !Ref PublicSubnet1

  # Route Tables
  PublicRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC

  DefaultPublicRoute:
    Type: AWS::EC2::Route
    DependsOn: InternetGatewayAttachment
    Properties:
      RouteTableId: !Ref PublicRouteTable
      DestinationCidrBlock: 0.0.0.0/0
      GatewayId: !Ref InternetGateway

  PublicSubnet1RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PublicRouteTable
      SubnetId: !Ref PublicSubnet1

  PublicSubnet2RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PublicRouteTable
      SubnetId: !Ref PublicSubnet2

  PrivateRouteTable1:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC

  DefaultPrivateRoute1:
    Type: AWS::EC2::Route
    Properties:
      RouteTableId: !Ref PrivateRouteTable1
      DestinationCidrBlock: 0.0.0.0/0
      NatGatewayId: !Ref NatGateway1

  PrivateSubnet1RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PrivateRouteTable1
      SubnetId: !Ref PrivateSubnet1

  PrivateSubnet2RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PrivateRouteTable1
      SubnetId: !Ref PrivateSubnet2

  # Security Groups
  ALBSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: ALB Security Group
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0

  ECSSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: ECS Security Group
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 8000
          ToPort: 8000
          SourceSecurityGroupId: !Ref ALBSecurityGroup

  DatabaseSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Database Security Group
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 5432
          ToPort: 5432
          SourceSecurityGroupId: !Ref ECSSecurityGroup

  # RDS Database
  DatabaseSubnetGroup:
    Type: AWS::RDS::DBSubnetGroup
    Properties:
      DBSubnetGroupDescription: Subnet group for RDS database
      SubnetIds:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2

  Database:
    Type: AWS::RDS::DBInstance
    DeletionPolicy: Snapshot
    Properties:
      DBInstanceIdentifier: !Sub ${Environment}-sperm-analysis-db
      DBName: sperm_analysis
      DBInstanceClass: db.r5.large
      AllocatedStorage: 100
      StorageType: gp2
      StorageEncrypted: true
      Engine: postgres
      EngineVersion: '15.4'
      MasterUsername: sperm_user
      MasterUserPassword: !Ref DatabasePassword
      VPCSecurityGroups:
        - !Ref DatabaseSecurityGroup
      DBSubnetGroupName: !Ref DatabaseSubnetGroup
      BackupRetentionPeriod: 30
      MultiAZ: true
      DeletionProtection: true

  # ElastiCache Redis
  CacheSubnetGroup:
    Type: AWS::ElastiCache::SubnetGroup
    Properties:
      Description: Cache subnet group
      SubnetIds:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2

  CacheSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Cache Security Group
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 6379
          ToPort: 6379
          SourceSecurityGroupId: !Ref ECSSecurityGroup

  RedisCache:
    Type: AWS::ElastiCache::ReplicationGroup
    Properties:
      ReplicationGroupId: !Sub ${Environment}-sperm-analysis-cache
      ReplicationGroupDescription: Redis cache for sperm analysis
      NodeType: cache.r6g.large
      Port: 6379
      ParameterGroupName: default.redis7
      NumCacheClusters: 2
      Engine: redis
      EngineVersion: '7.0'
      SubnetGroupName: !Ref CacheSubnetGroup
      SecurityGroupIds:
        - !Ref CacheSecurityGroup
      AtRestEncryptionEnabled: true
      TransitEncryptionEnabled: true

  # EFS for persistent storage
  EfsSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: EFS Security Group
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 2049
          ToPort: 2049
          SourceSecurityGroupId: !Ref ECSSecurityGroup

  EfsFileSystem:
    Type: AWS::EFS::FileSystem
    Properties:
      CreationToken: !Sub ${Environment}-sperm-analysis-efs
      Encrypted: true
      PerformanceMode: generalPurpose
      ThroughputMode: provisioned
      ProvisionedThroughputInMibps: 100

  EfsMountTarget1:
    Type: AWS::EFS::MountTarget
    Properties:
      FileSystemId: !Ref EfsFileSystem
      SubnetId: !Ref PrivateSubnet1
      SecurityGroups:
        - !Ref EfsSecurityGroup

  EfsMountTarget2:
    Type: AWS::EFS::MountTarget
    Properties:
      FileSystemId: !Ref EfsFileSystem
      SubnetId: !Ref PrivateSubnet2
      SecurityGroups:
        - !Ref EfsSecurityGroup

  # Application Load Balancer
  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: !Sub ${Environment}-sperm-analysis-alb
      Scheme: internet-facing
      Type: application
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2
      SecurityGroups:
        - !Ref ALBSecurityGroup

  ALBTargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Name: !Sub ${Environment}-sperm-analysis-tg
      Port: 8000
      Protocol: HTTP
      VpcId: !Ref VPC
      TargetType: ip
      HealthCheckPath: /health
      HealthCheckProtocol: HTTP
      HealthCheckIntervalSeconds: 30
      HealthCheckTimeoutSeconds: 5
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 3

  ALBListener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      DefaultActions:
        - Type: forward
          TargetGroupArn: !Ref ALBTargetGroup
      LoadBalancerArn: !Ref ApplicationLoadBalancer
      Port: 80
      Protocol: HTTP

  # ECS Cluster
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: !Sub ${Environment}-sperm-analysis
      CapacityProviders:
        - FARGATE
        - FARGATE_SPOT
      DefaultCapacityProviderStrategy:
        - CapacityProvider: FARGATE
          Weight: 1
        - CapacityProvider: FARGATE_SPOT
          Weight: 1

  # ECR Repository
  ECRRepository:
    Type: AWS::ECR::Repository
    Properties:
      RepositoryName: sperm-analysis
      ImageScanningConfiguration:
        ScanOnPush: true
      EncryptionConfiguration:
        EncryptionType: AES256

  # Secrets Manager
  ApplicationSecrets:
    Type: AWS::SecretsManager::Secret
    Properties:
      Name: !Sub ${Environment}-sperm-analysis-secrets
      Description: Application secrets for sperm analysis system
      GenerateSecretString:
        SecretStringTemplate: '{"username": "admin"}'
        GenerateStringKey: 'password'
        PasswordLength: 32
        ExcludeCharacters: '"@/\'

  # CloudWatch Log Group
  LogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub /ecs/${Environment}-sperm-analysis
      RetentionInDays: 30

Outputs:
  LoadBalancerURL:
    Description: Load Balancer URL
    Value: !Sub http://${ApplicationLoadBalancer.DNSName}
    Export:
      Name: !Sub ${Environment}-LoadBalancerURL

  DatabaseEndpoint:
    Description: RDS instance endpoint
    Value: !GetAtt Database.Endpoint.Address
    Export:
      Name: !Sub ${Environment}-DatabaseEndpoint

  RedisEndpoint:
    Description: Redis cluster endpoint
    Value: !GetAtt RedisCache.RedisEndpoint.Address
    Export:
      Name: !Sub ${Environment}-RedisEndpoint

  ECSClusterName:
    Description: ECS Cluster Name
    Value: !Ref ECSCluster
    Export:
      Name: !Sub ${Environment}-ECSCluster
```

### Deployment Script

Create `deploy/aws/deploy.sh`:

```bash
#!/bin/bash

# AWS Deployment Script for Sperm Analysis System
# Developer: Youssef Shitiwi

set -e

ENVIRONMENT=${1:-production}
AWS_REGION=${2:-us-west-2}
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "🧬 Deploying Sperm Analysis System to AWS"
echo "Environment: $ENVIRONMENT"
echo "Region: $AWS_REGION"
echo "Account: $AWS_ACCOUNT_ID"

# Build and push Docker image
echo "📦 Building and pushing Docker image..."
ECR_REPO="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/sperm-analysis"

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPO

# Build image
docker build -t sperm-analysis .
docker tag sperm-analysis:latest $ECR_REPO:latest
docker tag sperm-analysis:latest $ECR_REPO:$ENVIRONMENT-$(date +%Y%m%d-%H%M%S)

# Push images
docker push $ECR_REPO:latest
docker push $ECR_REPO:$ENVIRONMENT-$(date +%Y%m%d-%H%M%S)

# Deploy CloudFormation stack
echo "☁️ Deploying CloudFormation stack..."
aws cloudformation deploy \
  --template-file cloudformation.yaml \
  --stack-name $ENVIRONMENT-sperm-analysis \
  --parameter-overrides \
    Environment=$ENVIRONMENT \
    DatabasePassword=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25) \
  --capabilities CAPABILITY_IAM \
  --region $AWS_REGION

# Update ECS service
echo "🚀 Updating ECS service..."
sed "s/YOUR_ECR_REPO/$ECR_REPO/g" ecs-task-definition.json > ecs-task-definition-updated.json
sed -i "s/ACCOUNT_ID/$AWS_ACCOUNT_ID/g" ecs-task-definition-updated.json

aws ecs register-task-definition \
  --cli-input-json file://ecs-task-definition-updated.json \
  --region $AWS_REGION

aws ecs update-service \
  --cluster $ENVIRONMENT-sperm-analysis \
  --service sperm-analysis-service \
  --task-definition sperm-analysis-api \
  --region $AWS_REGION

echo "✅ Deployment completed successfully!"
echo "🌐 Application URL: $(aws cloudformation describe-stacks --stack-name $ENVIRONMENT-sperm-analysis --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerURL`].OutputValue' --output text --region $AWS_REGION)"
```

---

## 🌐 Google Cloud Platform

### Cloud Run Deployment

Create `deploy/gcp/cloudrun.yaml`:

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: sperm-analysis-api
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/execution-environment: gen2
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "4Gi"
        run.googleapis.com/cpu: "2"
        run.googleapis.com/max-scale: "10"
        run.googleapis.com/min-scale: "1"
    spec:
      containerConcurrency: 80
      containers:
      - image: gcr.io/PROJECT_ID/sperm-analysis:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: sperm-analysis-secrets
              key: database_url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: sperm-analysis-secrets
              key: redis_url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: sperm-analysis-secrets
              key: secret_key
        - name: GOOGLE_CLOUD_PROJECT
          value: "PROJECT_ID"
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
        volumeMounts:
        - name: gcs-fuse-csi-ephemeral
          mountPath: /app/data
      volumes:
      - name: gcs-fuse-csi-ephemeral
        csi:
          driver: gcs.csi.ofek.dev
          volumeAttributes:
            bucketName: "sperm-analysis-data-BUCKET_SUFFIX"
            mountOptions: "implicit-dirs"
```

### Terraform Configuration

Create `deploy/gcp/main.tf`:

```hcl
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "sql-component.googleapis.com",
    "sqladmin.googleapis.com",
    "storage.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "containerregistry.googleapis.com"
  ])
  
  service = each.value
  disable_on_destroy = false
}

# VPC Network
resource "google_compute_network" "sperm_analysis_vpc" {
  name                    = "${var.environment}-sperm-analysis-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "sperm_analysis_subnet" {
  name          = "${var.environment}-sperm-analysis-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.sperm_analysis_vpc.name
  
  secondary_ip_range {
    range_name    = "services-range"
    ip_cidr_range = "192.168.1.0/24"
  }
}

# Cloud SQL PostgreSQL
resource "google_sql_database_instance" "postgres" {
  name             = "${var.environment}-sperm-analysis-db"
  database_version = "POSTGRES_15"
  region           = var.region
  
  settings {
    tier      = "db-custom-2-7680"
    disk_size = 100
    disk_type = "PD_SSD"
    
    backup_configuration {
      enabled    = true
      start_time = "02:00"
      location   = var.region
      
      backup_retention_settings {
        retained_backups = 30
      }
    }
    
    database_flags {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements"
    }
    
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.sperm_analysis_vpc.id
      require_ssl     = true
    }
  }
  
  depends_on = [google_service_networking_connection.private_vpc_connection]
}

resource "google_sql_database" "sperm_analysis_db" {
  name     = "sperm_analysis"
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "sperm_user" {
  name     = "sperm_user"
  instance = google_sql_database_instance.postgres.name
  password = random_password.db_password.result
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}

# Private VPC connection for Cloud SQL
resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.sperm_analysis_vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

resource "google_compute_global_address" "private_ip_address" {
  name          = "${var.environment}-private-ip-address"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.sperm_analysis_vpc.id
}

# Memory Store Redis
resource "google_redis_instance" "cache" {
  name           = "${var.environment}-sperm-analysis-cache"
  tier           = "STANDARD_HA"
  memory_size_gb = 5
  region         = var.region
  
  authorized_network = google_compute_network.sperm_analysis_vpc.id
  
  redis_version     = "REDIS_7_0"
  display_name      = "Sperm Analysis Cache"
  
  auth_enabled = true
  
  depends_on = [google_service_networking_connection.private_vpc_connection]
}

# Cloud Storage buckets
resource "google_storage_bucket" "data_bucket" {
  name     = "${var.project_id}-sperm-analysis-data"
  location = var.region
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}

resource "google_storage_bucket" "models_bucket" {
  name     = "${var.project_id}-sperm-analysis-models"
  location = var.region
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
}

# Secret Manager secrets
resource "google_secret_manager_secret" "database_url" {
  secret_id = "${var.environment}-database-url"
  
  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }
}

resource "google_secret_manager_secret_version" "database_url" {
  secret      = google_secret_manager_secret.database_url.id
  secret_data = "postgresql://${google_sql_user.sperm_user.name}:${random_password.db_password.result}@${google_sql_database_instance.postgres.private_ip_address}:5432/${google_sql_database.sperm_analysis_db.name}"
}

resource "google_secret_manager_secret" "redis_url" {
  secret_id = "${var.environment}-redis-url"
  
  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }
}

resource "google_secret_manager_secret_version" "redis_url" {
  secret      = google_secret_manager_secret.redis_url.id
  secret_data = "redis://default:${google_redis_instance.cache.auth_string}@${google_redis_instance.cache.host}:${google_redis_instance.cache.port}/0"
}

resource "google_secret_manager_secret" "secret_key" {
  secret_id = "${var.environment}-secret-key"
  
  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }
}

resource "google_secret_manager_secret_version" "secret_key" {
  secret      = google_secret_manager_secret.secret_key.id
  secret_data = random_password.secret_key.result
}

resource "random_password" "secret_key" {
  length  = 50
  special = true
}

# Cloud Run service
resource "google_cloud_run_service" "sperm_analysis_api" {
  name     = "${var.environment}-sperm-analysis-api"
  location = var.region
  
  template {
    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale"        = "10"
        "autoscaling.knative.dev/minScale"        = "1"
        "run.googleapis.com/execution-environment" = "gen2"
        "run.googleapis.com/cpu-throttling"        = "false"
      }
    }
    
    spec {
      container_concurrency = 80
      
      containers {
        image = "gcr.io/${var.project_id}/sperm-analysis:latest"
        
        ports {
          container_port = 8000
        }
        
        resources {
          limits = {
            cpu    = "2"
            memory = "4Gi"
          }
        }
        
        env {
          name = "DATABASE_URL"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.database_url.secret_id
              key  = "latest"
            }
          }
        }
        
        env {
          name = "REDIS_URL"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.redis_url.secret_id
              key  = "latest"
            }
          }
        }
        
        env {
          name = "SECRET_KEY"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.secret_key.secret_id
              key  = "latest"
            }
          }
        }
        
        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }
        
        env {
          name  = "STORAGE_BUCKET"
          value = google_storage_bucket.data_bucket.name
        }
        
        env {
          name  = "MODELS_BUCKET"
          value = google_storage_bucket.models_bucket.name
        }
      }
    }
  }
  
  traffic {
    percent         = 100
    latest_revision = true
  }
  
  depends_on = [google_project_service.required_apis]
}

# IAM policy for Cloud Run
resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.sperm_analysis_api.name
  location = google_cloud_run_service.sperm_analysis_api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Outputs
output "cloud_run_url" {
  value = google_cloud_run_service.sperm_analysis_api.status[0].url
}

output "database_ip" {
  value = google_sql_database_instance.postgres.private_ip_address
}

output "redis_host" {
  value = google_redis_instance.cache.host
}

output "data_bucket" {
  value = google_storage_bucket.data_bucket.name
}
```

---

## 🔷 Microsoft Azure

### Azure Container Instances

Create `deploy/azure/aci-deployment.yaml`:

```yaml
apiVersion: '2019-12-01'
location: eastus
name: sperm-analysis-system
properties:
  containers:
  - name: sperm-analysis-api
    properties:
      image: spermanalysisregistry.azurecr.io/sperm-analysis:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 4
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: DATABASE_URL
        secureValue: postgresql://username@server:password@hostname:5432/database
      - name: REDIS_URL
        secureValue: rediss://:password@hostname:6380/0
      - name: SECRET_KEY
        secureValue: your-secret-key
      - name: AZURE_STORAGE_ACCOUNT
        value: spermanalysisstorage
      - name: AZURE_STORAGE_CONTAINER
        value: data
      volumeMounts:
      - name: azure-file-volume
        mountPath: /app/data
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8000
    dnsNameLabel: sperm-analysis-api
  volumes:
  - name: azure-file-volume
    azureFile:
      shareName: sperm-analysis-data
      storageAccountName: spermanalysisstorage
      storageAccountKey: your-storage-key
tags:
  Environment: production
  Developer: Youssef Shitiwi
  Application: SpermAnalysis
```

### ARM Template

Create `deploy/azure/azuredeploy.json`:

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "environment": {
      "type": "string",
      "defaultValue": "production",
      "allowedValues": ["development", "staging", "production"]
    },
    "location": {
      "type": "string",
      "defaultValue": "[resourceGroup().location]"
    },
    "databaseAdminPassword": {
      "type": "securestring",
      "minLength": 8
    }
  },
  "variables": {
    "resourcePrefix": "[concat(parameters('environment'), '-sperm-analysis')]",
    "vnetName": "[concat(variables('resourcePrefix'), '-vnet')]",
    "subnetName": "[concat(variables('resourcePrefix'), '-subnet')]",
    "storageAccountName": "[replace(concat(variables('resourcePrefix'), 'storage'), '-', '')]",
    "keyVaultName": "[concat(variables('resourcePrefix'), '-kv')]",
    "postgresServerName": "[concat(variables('resourcePrefix'), '-postgres')]",
    "redisName": "[concat(variables('resourcePrefix'), '-redis')]",
    "containerRegistryName": "[replace(concat(variables('resourcePrefix'), 'registry'), '-', '')]"
  },
  "resources": [
    {
      "type": "Microsoft.Network/virtualNetworks",
      "apiVersion": "2020-11-01",
      "name": "[variables('vnetName')]",
      "location": "[parameters('location')]",
      "properties": {
        "addressSpace": {
          "addressPrefixes": ["10.0.0.0/16"]
        },
        "subnets": [
          {
            "name": "[variables('subnetName')]",
            "properties": {
              "addressPrefix": "10.0.1.0/24",
              "serviceEndpoints": [
                {
                  "service": "Microsoft.Storage"
                },
                {
                  "service": "Microsoft.Sql"
                }
              ]
            }
          }
        ]
      }
    },
    {
      "type": "Microsoft.Storage/storageAccounts",
      "apiVersion": "2021-06-01",
      "name": "[variables('storageAccountName')]",
      "location": "[parameters('location')]",
      "sku": {
        "name": "Standard_LRS"
      },
      "kind": "StorageV2",
      "properties": {
        "supportsHttpsTrafficOnly": true,
        "encryption": {
          "services": {
            "file": {
              "enabled": true
            },
            "blob": {
              "enabled": true
            }
          },
          "keySource": "Microsoft.Storage"
        }
      }
    },
    {
      "type": "Microsoft.KeyVault/vaults",
      "apiVersion": "2021-06-01-preview",
      "name": "[variables('keyVaultName')]",
      "location": "[parameters('location')]",
      "properties": {
        "tenantId": "[subscription().tenantId]",
        "sku": {
          "family": "A",
          "name": "standard"
        },
        "accessPolicies": [],
        "enabledForDeployment": false,
        "enabledForDiskEncryption": false,
        "enabledForTemplateDeployment": true,
        "enableSoftDelete": true,
        "softDeleteRetentionInDays": 90
      }
    },
    {
      "type": "Microsoft.DBforPostgreSQL/flexibleServers",
      "apiVersion": "2021-06-01",
      "name": "[variables('postgresServerName')]",
      "location": "[parameters('location')]",
      "sku": {
        "name": "Standard_D2s_v3",
        "tier": "GeneralPurpose"
      },
      "properties": {
        "version": "15",
        "administratorLogin": "sperm_admin",
        "administratorLoginPassword": "[parameters('databaseAdminPassword')]",
        "storage": {
          "storageSizeGB": 128
        },
        "backup": {
          "backupRetentionDays": 30,
          "geoRedundantBackup": "Enabled"
        },
        "highAvailability": {
          "mode": "ZoneRedundant"
        }
      }
    },
    {
      "type": "Microsoft.Cache/Redis",
      "apiVersion": "2020-06-01",
      "name": "[variables('redisName')]",
      "location": "[parameters('location')]",
      "properties": {
        "sku": {
          "name": "Premium",
          "family": "P",
          "capacity": 1
        },
        "enableNonSslPort": false,
        "minimumTlsVersion": "1.2",
        "redisConfiguration": {
          "maxmemory-policy": "allkeys-lru"
        }
      }
    },
    {
      "type": "Microsoft.ContainerRegistry/registries",
      "apiVersion": "2021-06-01-preview",
      "name": "[variables('containerRegistryName')]",
      "location": "[parameters('location')]",
      "sku": {
        "name": "Premium"
      },
      "properties": {
        "adminUserEnabled": true
      }
    }
  ],
  "outputs": {
    "storageAccountName": {
      "type": "string",
      "value": "[variables('storageAccountName')]"
    },
    "postgresServerName": {
      "type": "string",
      "value": "[variables('postgresServerName')]"
    },
    "redisHostName": {
      "type": "string",
      "value": "[reference(resourceId('Microsoft.Cache/Redis', variables('redisName'))).hostName]"
    },
    "keyVaultName": {
      "type": "string",
      "value": "[variables('keyVaultName')]"
    }
  }
}
```

---

## ⚓ Kubernetes Deployment

### Kubernetes Manifests

Create `deploy/k8s/namespace.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: sperm-analysis
  labels:
    name: sperm-analysis
    developer: youssef-shitiwi
```

Create `deploy/k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sperm-analysis-api
  namespace: sperm-analysis
  labels:
    app: sperm-analysis-api
    developer: youssef-shitiwi
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sperm-analysis-api
  template:
    metadata:
      labels:
        app: sperm-analysis-api
    spec:
      containers:
      - name: api
        image: sperm-analysis:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: sperm-analysis-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: sperm-analysis-secrets
              key: redis-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: sperm-analysis-secrets
              key: secret-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: sperm-analysis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: sperm-analysis-service
  namespace: sperm-analysis
spec:
  selector:
    app: sperm-analysis-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sperm-analysis-ingress
  namespace: sperm-analysis
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/proxy-body-size: "500m"
spec:
  tls:
  - hosts:
    - api.spermanalysis.com
    secretName: sperm-analysis-tls
  rules:
  - host: api.spermanalysis.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sperm-analysis-service
            port:
              number: 80
```

### Helm Chart

Create `deploy/helm/sperm-analysis/Chart.yaml`:

```yaml
apiVersion: v2
name: sperm-analysis
description: A Helm chart for Sperm Analysis System by Youssef Shitiwi
type: application
version: 1.0.0
appVersion: "1.0.0"
keywords:
  - sperm-analysis
  - ai
  - casa
  - medical
home: https://github.com/youssefshitiwi/sperm-analysis
sources:
  - https://github.com/youssefshitiwi/sperm-analysis
maintainers:
  - name: Youssef Shitiwi
    email: youssef.shitiwi@example.com
```

Create `deploy/helm/sperm-analysis/values.yaml`:

```yaml
# Default values for sperm-analysis
replicaCount: 3

image:
  repository: sperm-analysis
  pullPolicy: IfNotPresent
  tag: "latest"

nameOverride: ""
fullnameOverride: ""

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/proxy-body-size: "500m"
  hosts:
    - host: api.spermanalysis.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: sperm-analysis-tls
      hosts:
        - api.spermanalysis.com

resources:
  limits:
    cpu: 2
    memory: 4Gi
  requests:
    cpu: 1
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
  targetMemoryUtilizationPercentage: 80

persistence:
  enabled: true
  storageClass: "fast-ssd"
  accessMode: ReadWriteMany
  size: 100Gi

postgresql:
  enabled: true
  auth:
    postgresPassword: "changeme"
    database: "sperm_analysis"
  primary:
    persistence:
      enabled: true
      size: 100Gi

redis:
  enabled: true
  auth:
    enabled: true
    password: "changeme"
  master:
    persistence:
      enabled: true
      size: 10Gi

secrets:
  secretKey: "your-secret-key"
  databaseUrl: "postgresql://user:pass@host:5432/db"
  redisUrl: "redis://:pass@host:6379/0"

config:
  environment: "production"
  debug: false
  logLevel: "INFO"
  maxWorkers: 4
  uploadMaxSize: "500MB"

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s
```

---

## ⚡ Serverless Options

### AWS Lambda with API Gateway

Create `deploy/serverless/serverless.yml`:

```yaml
service: sperm-analysis-serverless

frameworkVersion: '3'

provider:
  name: aws
  runtime: python3.9
  region: us-west-2
  stage: ${opt:stage, 'prod'}
  memorySize: 3008
  timeout: 900
  
  environment:
    DATABASE_URL: ${ssm:/sperm-analysis/${self:provider.stage}/database-url}
    REDIS_URL: ${ssm:/sperm-analysis/${self:provider.stage}/redis-url}
    SECRET_KEY: ${ssm:/sperm-analysis/${self:provider.stage}/secret-key}
    S3_BUCKET: ${cf:sperm-analysis-infrastructure-${self:provider.stage}.DataBucket}
    
  iamRoleStatements:
    - Effect: Allow
      Action:
        - s3:GetObject
        - s3:PutObject
        - s3:DeleteObject
      Resource: "arn:aws:s3:::${self:provider.environment.S3_BUCKET}/*"
    - Effect: Allow
      Action:
        - ssm:GetParameter
        - ssm:GetParameters
      Resource: "arn:aws:ssm:${self:provider.region}:*:parameter/sperm-analysis/${self:provider.stage}/*"

functions:
  api:
    handler: serverless_wsgi.handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
          cors: true
      - http:
          path: /
          method: ANY
          cors: true
    layers:
      - arn:aws:lambda:us-west-2:889247394523:layer:scipy:2
      - arn:aws:lambda:us-west-2:889247394523:layer:opencv:1

  processVideo:
    handler: handlers.process_video
    events:
      - s3:
          bucket: ${self:provider.environment.S3_BUCKET}
          event: s3:ObjectCreated:*
          rules:
            - prefix: uploads/
            - suffix: .mp4
    timeout: 900
    memorySize: 3008

plugins:
  - serverless-wsgi
  - serverless-python-requirements

custom:
  wsgi:
    app: app.main
    packRequirements: false
  pythonRequirements:
    dockerizePip: true
    layer: true
    slim: true
    strip: false
```

---

## 🔄 Multi-Cloud Strategy

### Terraform Multi-Cloud

Create `deploy/multi-cloud/main.tf`:

```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

# AWS Configuration
provider "aws" {
  region = var.aws_region
  alias  = "primary"
}

# GCP Configuration
provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
  alias   = "secondary"
}

# Azure Configuration
provider "azurerm" {
  features {}
  alias = "tertiary"
}

variable "aws_region" {
  default = "us-west-2"
}

variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "gcp_region" {
  default = "us-central1"
}

variable "environment" {
  default = "production"
}

# AWS Primary Deployment
module "aws_deployment" {
  source = "./modules/aws"
  
  providers = {
    aws = aws.primary
  }
  
  environment = var.environment
  region      = var.aws_region
}

# GCP Secondary Deployment
module "gcp_deployment" {
  source = "./modules/gcp"
  
  providers = {
    google = google.secondary
  }
  
  environment = var.environment
  project_id  = var.gcp_project_id
  region      = var.gcp_region
}

# Route 53 for global DNS
resource "aws_route53_zone" "main" {
  provider = aws.primary
  name     = "spermanalysis.com"
}

resource "aws_route53_record" "api_primary" {
  provider = aws.primary
  zone_id  = aws_route53_zone.main.zone_id
  name     = "api"
  type     = "A"
  
  set_identifier = "primary"
  
  alias {
    name                   = module.aws_deployment.load_balancer_dns
    zone_id               = module.aws_deployment.load_balancer_zone_id
    evaluate_target_health = true
  }
  
  failover_routing_policy {
    type = "PRIMARY"
  }
}

resource "aws_route53_record" "api_secondary" {
  provider = aws.primary
  zone_id  = aws_route53_zone.main.zone_id
  name     = "api"
  type     = "CNAME"
  ttl      = 300
  
  set_identifier = "secondary"
  records        = [module.gcp_deployment.cloud_run_url]
  
  failover_routing_policy {
    type = "SECONDARY"
  }
}

# Health checks
resource "aws_route53_health_check" "primary" {
  provider                            = aws.primary
  fqdn                               = module.aws_deployment.load_balancer_dns
  port                               = 443
  type                               = "HTTPS"
  resource_path                      = "/health"
  failure_threshold                  = 3
  request_interval                   = 30
  cloudwatch_alarm_region           = var.aws_region
  cloudwatch_alarm_name             = "sperm-analysis-primary-health"
  insufficient_data_health_status   = "Failure"
}

# Outputs
output "primary_endpoint" {
  value = "https://${aws_route53_record.api_primary.fqdn}"
}

output "secondary_endpoint" {
  value = module.gcp_deployment.cloud_run_url
}

output "global_endpoint" {
  value = "https://api.spermanalysis.com"
}
```

---

## 📊 Performance Optimization

### CDN Configuration

Create `deploy/cdn/cloudfront.tf`:

```hcl
resource "aws_cloudfront_distribution" "sperm_analysis_cdn" {
  origin {
    domain_name = aws_route53_record.api_primary.fqdn
    origin_id   = "sperm-analysis-origin"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  enabled = true
  
  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "sperm-analysis-origin"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"
    
    forwarded_values {
      query_string = true
      headers      = ["Authorization", "Content-Type"]
      
      cookies {
        forward = "none"
      }
    }
    
    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 0
  }
  
  # Cache static assets
  ordered_cache_behavior {
    path_pattern     = "/static/*"
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "sperm-analysis-origin"
    compress         = true
    
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 31536000
    default_ttl            = 31536000
    max_ttl                = 31536000
  }
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    acm_certificate_arn = aws_acm_certificate.cert.arn
    ssl_support_method  = "sni-only"
  }
  
  web_acl_id = aws_wafv2_web_acl.sperm_analysis_waf.arn
  
  tags = {
    Name        = "sperm-analysis-cdn"
    Environment = var.environment
    Developer   = "Youssef Shitiwi"
  }
}
```

---

This comprehensive cloud deployment guide provides production-ready configurations for all major cloud platforms, with emphasis on scalability, security, and performance optimization. All deployments maintain the high standards established by developer **Youssef Shitiwi**.

For specific deployment requirements or advanced customization, refer to the individual platform documentation or contact the development team.