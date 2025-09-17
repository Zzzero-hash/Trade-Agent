variable "environment" {
  description = "Environment name (e.g., prod, staging, dev)"
  type        = string
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs"
  type        = list(string)
}

variable "public_subnet_ids" {
  description = "List of public subnet IDs"
  type        = list(string)
}

variable "cluster_security_group_id" {
  description = "Security group ID for EKS cluster"
  type        = string
}

variable "nodes_security_group_id" {
  description = "Security group ID for EKS nodes"
  type        = string
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
      instance_types = ["t3.medium"]
      ami_type       = "AL2_x86_64"
      disk_size      = 50
      desired_size   = 2
      max_size       = 10
      min_size       = 1
      labels         = {}
      taints         = []
    }
  }
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default     = {}
}