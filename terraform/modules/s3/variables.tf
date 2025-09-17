variable "environment" {
  description = "Environment name (e.g., prod, staging, dev)"
  type        = string
}

variable "enable_cross_region_replication" {
  description = "Enable cross-region replication for compliance bucket"
  type        = bool
  default     = false
}

variable "replication_bucket_name" {
  description = "Name of the destination bucket for cross-region replication"
  type        = string
  default     = ""
}

variable "replication_kms_key_id" {
  description = "KMS key ID for replication encryption"
  type        = string
  default     = ""
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default     = {}
}