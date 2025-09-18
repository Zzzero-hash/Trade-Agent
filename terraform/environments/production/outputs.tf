# Production Environment Outputs

# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = module.vpc.private_subnet_ids
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = module.vpc.public_subnet_ids
}

output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = module.vpc.database_subnet_ids
}

# EKS Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_id
}

# RDS Outputs
output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.db_instance_endpoint
}

output "rds_port" {
  description = "RDS instance port"
  value       = module.rds.db_instance_port
}

output "rds_database_name" {
  description = "RDS instance database name"
  value       = module.rds.db_instance_name
}

output "rds_username" {
  description = "RDS instance master username"
  value       = module.rds.db_instance_username
  sensitive   = true
}

output "rds_password_secret_arn" {
  description = "ARN of the secret containing the RDS password"
  value       = module.rds.db_password_secret_arn
}

output "rds_read_replica_endpoint" {
  description = "Read replica endpoint"
  value       = module.rds.read_replica_endpoint
}

# ElastiCache Outputs
output "redis_primary_endpoint" {
  description = "Address of the endpoint for the primary node in the replication group"
  value       = module.elasticache.primary_endpoint_address
}

output "redis_reader_endpoint" {
  description = "Address of the endpoint for the reader node in the replication group"
  value       = module.elasticache.reader_endpoint_address
}

output "redis_port" {
  description = "Port number on which the cache nodes accept connections"
  value       = module.elasticache.port
}

output "redis_auth_token_secret_arn" {
  description = "ARN of the secret containing the Redis auth token"
  value       = module.elasticache.auth_token_secret_arn
}

# S3 Outputs
output "data_storage_bucket_name" {
  description = "Name of the data storage bucket"
  value       = module.s3.data_storage_bucket_id
}

output "model_artifacts_bucket_name" {
  description = "Name of the model artifacts bucket"
  value       = module.s3.model_artifacts_bucket_id
}

output "backups_bucket_name" {
  description = "Name of the backups bucket"
  value       = module.s3.backups_bucket_id
}

output "compliance_bucket_name" {
  description = "Name of the compliance bucket"
  value       = module.s3.compliance_bucket_id
}

# Security Group Outputs
output "alb_security_group_id" {
  description = "ID of the ALB security group"
  value       = module.security_groups.alb_security_group_id
}

output "eks_nodes_security_group_id" {
  description = "ID of the EKS nodes security group"
  value       = module.security_groups.eks_nodes_security_group_id
}