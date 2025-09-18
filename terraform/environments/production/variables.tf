# Production Environment Variables

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "owner" {
  description = "Owner of the resources"
  type        = string
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
}

# EKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "List of CIDR blocks that can access the Amazon EKS public API server endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "node_group_ssh_key" {
  description = "EC2 Key Pair name for SSH access to worker nodes"
  type        = string
  default     = null
}

variable "node_groups" {
  description = "Map of EKS node group configurations"
  type = map(object({
    capacity_type  = string
    instance_types = list(string)
    ami_type       = string
    disk_size      = number
    desired_size   = number
    max_size       = number
    min_size       = number
    labels         = map(string)
    taints = list(object({
      key    = string
      value  = string
      effect = string
    }))
  }))
  default = {
    general = {
      capacity_type  = "ON_DEMAND"
      instance_types = ["m5.large"]
      ami_type       = "AL2_x86_64"
      disk_size      = 50
      desired_size   = 3
      max_size       = 10
      min_size       = 2
      labels = {
        role = "general"
      }
      taints = []
    }
    ml_workload = {
      capacity_type  = "ON_DEMAND"
      instance_types = ["c5.2xlarge"]
      ami_type       = "AL2_x86_64"
      disk_size      = 100
      desired_size   = 2
      max_size       = 5
      min_size       = 1
      labels = {
        role = "ml-workload"
      }
      taints = [
        {
          key    = "ml-workload"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
}

# RDS Configuration
variable "postgres_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15.4"
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.xlarge"
}

variable "rds_allocated_storage" {
  description = "Initial allocated storage in GB"
  type        = number
  default     = 200
}

variable "rds_max_allocated_storage" {
  description = "Maximum allocated storage in GB for autoscaling"
  type        = number
  default     = 2000
}

variable "database_name" {
  description = "Name of the database"
  type        = string
  default     = "trading_platform"
}

variable "rds_master_username" {
  description = "Master username for the database"
  type        = string
  default     = "postgres"
}

variable "rds_multi_az" {
  description = "Enable Multi-AZ deployment"
  type        = bool
  default     = true
}

variable "rds_backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
}

variable "rds_deletion_protection" {
  description = "Enable deletion protection"
  type        = bool
  default     = true
}

variable "rds_create_read_replica" {
  description = "Create read replica"
  type        = bool
  default     = true
}

variable "rds_read_replica_instance_class" {
  description = "Instance class for read replica"
  type        = string
  default     = "db.r6g.large"
}

# ElastiCache Configuration
variable "elasticache_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.r7g.xlarge"
}

variable "redis_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.0"
}

variable "elasticache_num_cache_clusters" {
  description = "Number of cache clusters (nodes) in the replication group"
  type        = number
  default     = 3
}

variable "elasticache_automatic_failover_enabled" {
  description = "Enable automatic failover"
  type        = bool
  default     = true
}

variable "elasticache_multi_az_enabled" {
  description = "Enable Multi-AZ"
  type        = bool
  default     = true
}

variable "elasticache_auth_token_enabled" {
  description = "Enable auth token for Redis"
  type        = bool
  default     = true
}

variable "elasticache_snapshot_retention_limit" {
  description = "Number of days to retain automatic snapshots"
  type        = number
  default     = 7
}

# S3 Configuration
variable "s3_enable_cross_region_replication" {
  description = "Enable cross-region replication for compliance bucket"
  type        = bool
  default     = true
}

variable "s3_replication_bucket_name" {
  description = "Name of the destination bucket for cross-region replication"
  type        = string
  default     = ""
}

variable "s3_replication_kms_key_id" {
  description = "KMS key ID for replication encryption"
  type        = string
  default     = ""
}

# Monitoring Configuration
variable "alarm_actions" {
  description = "List of ARNs to notify when alarm triggers"
  type        = list(string)
  default     = []
}