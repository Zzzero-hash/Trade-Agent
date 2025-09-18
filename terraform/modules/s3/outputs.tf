output "data_storage_bucket_id" {
  description = "ID of the data storage bucket"
  value       = aws_s3_bucket.data_storage.id
}

output "data_storage_bucket_arn" {
  description = "ARN of the data storage bucket"
  value       = aws_s3_bucket.data_storage.arn
}

output "model_artifacts_bucket_id" {
  description = "ID of the model artifacts bucket"
  value       = aws_s3_bucket.model_artifacts.id
}

output "model_artifacts_bucket_arn" {
  description = "ARN of the model artifacts bucket"
  value       = aws_s3_bucket.model_artifacts.arn
}

output "backups_bucket_id" {
  description = "ID of the backups bucket"
  value       = aws_s3_bucket.backups.id
}

output "backups_bucket_arn" {
  description = "ARN of the backups bucket"
  value       = aws_s3_bucket.backups.arn
}

output "compliance_bucket_id" {
  description = "ID of the compliance bucket"
  value       = aws_s3_bucket.compliance.id
}

output "compliance_bucket_arn" {
  description = "ARN of the compliance bucket"
  value       = aws_s3_bucket.compliance.arn
}

output "kms_key_id" {
  description = "KMS key ID for S3 encryption"
  value       = aws_kms_key.s3.key_id
}

output "kms_key_arn" {
  description = "KMS key ARN for S3 encryption"
  value       = aws_kms_key.s3.arn
}