# S3 Buckets with Versioning and Lifecycle Policies

# KMS Key for S3 encryption
resource "aws_kms_key" "s3" {
  description             = "S3 encryption key"
  deletion_window_in_days = 7

  tags = merge(var.common_tags, {
    Name = "${var.environment}-s3-key"
  })
}

resource "aws_kms_alias" "s3" {
  name          = "alias/${var.environment}-s3"
  target_key_id = aws_kms_key.s3.key_id
}

# Data Storage Bucket
resource "aws_s3_bucket" "data_storage" {
  bucket = "${var.environment}-trading-platform-data-${random_id.bucket_suffix.hex}"

  tags = merge(var.common_tags, {
    Name        = "${var.environment}-data-storage"
    Purpose     = "Data Storage"
    Environment = var.environment
  })
}

resource "aws_s3_bucket_versioning" "data_storage" {
  bucket = aws_s3_bucket.data_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_storage" {
  bucket = aws_s3_bucket.data_storage.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "data_storage" {
  bucket = aws_s3_bucket.data_storage.id

  rule {
    id     = "data_lifecycle"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 2555  # 7 years
    }
  }
}

# Model Artifacts Bucket
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${var.environment}-trading-platform-models-${random_id.bucket_suffix.hex}"

  tags = merge(var.common_tags, {
    Name        = "${var.environment}-model-artifacts"
    Purpose     = "Model Artifacts"
    Environment = var.environment
  })
}

resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  rule {
    id     = "model_lifecycle"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }
}

# Backup Bucket
resource "aws_s3_bucket" "backups" {
  bucket = "${var.environment}-trading-platform-backups-${random_id.bucket_suffix.hex}"

  tags = merge(var.common_tags, {
    Name        = "${var.environment}-backups"
    Purpose     = "Backups"
    Environment = var.environment
  })
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    id     = "backup_lifecycle"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }

    noncurrent_version_expiration {
      noncurrent_days = 2555  # 7 years for compliance
    }
  }
}

# Compliance Documents Bucket
resource "aws_s3_bucket" "compliance" {
  bucket = "${var.environment}-trading-platform-compliance-${random_id.bucket_suffix.hex}"

  tags = merge(var.common_tags, {
    Name        = "${var.environment}-compliance"
    Purpose     = "Compliance Documents"
    Environment = var.environment
  })
}

resource "aws_s3_bucket_versioning" "compliance" {
  bucket = aws_s3_bucket.compliance.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "compliance" {
  bucket = aws_s3_bucket.compliance.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

# Compliance bucket has no lifecycle policy - documents must be retained indefinitely

# Random suffix for bucket names
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Block public access for all buckets
resource "aws_s3_bucket_public_access_block" "data_storage" {
  bucket = aws_s3_bucket.data_storage.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "backups" {
  bucket = aws_s3_bucket.backups.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "compliance" {
  bucket = aws_s3_bucket.compliance.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Bucket notifications for monitoring
resource "aws_s3_bucket_notification" "data_storage" {
  bucket = aws_s3_bucket.data_storage.id

  cloudwatch_configuration {
    events = ["s3:ObjectCreated:*", "s3:ObjectRemoved:*"]
  }
}

# Cross-region replication for critical buckets (if enabled)
resource "aws_s3_bucket_replication_configuration" "compliance" {
  count = var.enable_cross_region_replication ? 1 : 0

  role   = aws_iam_role.replication[0].arn
  bucket = aws_s3_bucket.compliance.id

  rule {
    id     = "compliance_replication"
    status = "Enabled"

    destination {
      bucket        = "arn:aws:s3:::${var.replication_bucket_name}"
      storage_class = "STANDARD_IA"

      encryption_configuration {
        replica_kms_key_id = var.replication_kms_key_id
      }
    }
  }

  depends_on = [aws_s3_bucket_versioning.compliance]
}

# IAM role for replication
resource "aws_iam_role" "replication" {
  count = var.enable_cross_region_replication ? 1 : 0

  name = "${var.environment}-s3-replication-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
      }
    ]
  })

  tags = var.common_tags
}

resource "aws_iam_policy" "replication" {
  count = var.enable_cross_region_replication ? 1 : 0

  name = "${var.environment}-s3-replication-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl"
        ]
        Resource = "${aws_s3_bucket.compliance.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.compliance.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete"
        ]
        Resource = "arn:aws:s3:::${var.replication_bucket_name}/*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "replication" {
  count = var.enable_cross_region_replication ? 1 : 0

  role       = aws_iam_role.replication[0].name
  policy_arn = aws_iam_policy.replication[0].arn
}