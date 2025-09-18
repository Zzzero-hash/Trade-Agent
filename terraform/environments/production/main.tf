# Production Environment Infrastructure

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }

  backend "s3" {
    # Backend configuration will be provided via backend config file
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.common_tags
  }
}

# Local values
locals {
  environment = "production"
  
  common_tags = {
    Environment   = local.environment
    Project      = "AI Trading Platform"
    ManagedBy    = "Terraform"
    Owner        = var.owner
    CostCenter   = var.cost_center
  }

  availability_zones = data.aws_availability_zones.available.names

  # Network configuration
  vpc_cidr = "10.0.0.0/16"
  
  public_subnet_cidrs = [
    "10.0.1.0/24",
    "10.0.2.0/24",
    "10.0.3.0/24"
  ]
  
  private_subnet_cidrs = [
    "10.0.11.0/24",
    "10.0.12.0/24",
    "10.0.13.0/24"
  ]
  
  database_subnet_cidrs = [
    "10.0.21.0/24",
    "10.0.22.0/24",
    "10.0.23.0/24"
  ]
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC Module
module "vpc" {
  source = "../../modules/vpc"

  environment               = local.environment
  vpc_cidr                 = local.vpc_cidr
  availability_zones       = local.availability_zones
  public_subnet_cidrs      = local.public_subnet_cidrs
  private_subnet_cidrs     = local.private_subnet_cidrs
  database_subnet_cidrs    = local.database_subnet_cidrs
  common_tags              = local.common_tags
}

# Security Groups Module
module "security_groups" {
  source = "../../modules/security-groups"

  environment      = local.environment
  vpc_id          = module.vpc.vpc_id
  vpc_cidr_block  = module.vpc.vpc_cidr_block
  common_tags     = local.common_tags
}

# EKS Module
module "eks" {
  source = "../../modules/eks"

  environment                           = local.environment
  kubernetes_version                    = var.kubernetes_version
  private_subnet_ids                    = module.vpc.private_subnet_ids
  public_subnet_ids                     = module.vpc.public_subnet_ids
  cluster_security_group_id             = module.security_groups.eks_cluster_security_group_id
  nodes_security_group_id               = module.security_groups.eks_nodes_security_group_id
  cluster_endpoint_public_access_cidrs  = var.cluster_endpoint_public_access_cidrs
  node_group_ssh_key                    = var.node_group_ssh_key
  node_groups                           = var.node_groups
  common_tags                           = local.common_tags
}

# RDS Module
module "rds" {
  source = "../../modules/rds"

  environment           = local.environment
  database_subnet_ids   = module.vpc.database_subnet_ids
  security_group_id     = module.security_groups.rds_security_group_id
  postgres_version      = var.postgres_version
  instance_class        = var.rds_instance_class
  allocated_storage     = var.rds_allocated_storage
  max_allocated_storage = var.rds_max_allocated_storage
  database_name         = var.database_name
  master_username       = var.rds_master_username
  multi_az              = var.rds_multi_az
  backup_retention_period = var.rds_backup_retention_period
  deletion_protection   = var.rds_deletion_protection
  create_read_replica   = var.rds_create_read_replica
  read_replica_instance_class = var.rds_read_replica_instance_class
  common_tags           = local.common_tags
}

# ElastiCache Module
module "elasticache" {
  source = "../../modules/elasticache"

  environment                = local.environment
  private_subnet_ids         = module.vpc.private_subnet_ids
  security_group_id          = module.security_groups.elasticache_security_group_id
  node_type                  = var.elasticache_node_type
  redis_version              = var.redis_version
  num_cache_clusters         = var.elasticache_num_cache_clusters
  automatic_failover_enabled = var.elasticache_automatic_failover_enabled
  multi_az_enabled           = var.elasticache_multi_az_enabled
  auth_token_enabled         = var.elasticache_auth_token_enabled
  snapshot_retention_limit   = var.elasticache_snapshot_retention_limit
  alarm_actions              = var.alarm_actions
  common_tags                = local.common_tags
}

# S3 Module
module "s3" {
  source = "../../modules/s3"

  environment                     = local.environment
  enable_cross_region_replication = var.s3_enable_cross_region_replication
  replication_bucket_name         = var.s3_replication_bucket_name
  replication_kms_key_id          = var.s3_replication_kms_key_id
  common_tags                     = local.common_tags
}