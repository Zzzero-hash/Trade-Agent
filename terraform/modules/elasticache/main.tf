# ElastiCache Redis Cluster with Failover

# ElastiCache Subnet Group
resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.environment}-redis-subnet-group"
  subnet_ids = var.private_subnet_ids

  tags = var.common_tags
}

# ElastiCache Parameter Group
resource "aws_elasticache_parameter_group" "main" {
  family = "redis7.x"
  name   = "${var.environment}-redis-params"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  parameter {
    name  = "timeout"
    value = "300"
  }

  parameter {
    name  = "tcp-keepalive"
    value = "300"
  }

  tags = var.common_tags
}

# KMS Key for ElastiCache encryption
resource "aws_kms_key" "elasticache" {
  description             = "ElastiCache encryption key"
  deletion_window_in_days = 7

  tags = merge(var.common_tags, {
    Name = "${var.environment}-elasticache-key"
  })
}

resource "aws_kms_alias" "elasticache" {
  name          = "alias/${var.environment}-elasticache"
  target_key_id = aws_kms_key.elasticache.key_id
}

# ElastiCache Replication Group (Redis Cluster)
resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "${var.environment}-redis"
  description                = "Redis cluster for ${var.environment} environment"

  # Node configuration
  node_type               = var.node_type
  port                    = 6379
  parameter_group_name    = aws_elasticache_parameter_group.main.name

  # Cluster configuration
  num_cache_clusters      = var.num_cache_clusters
  automatic_failover_enabled = var.automatic_failover_enabled
  multi_az_enabled        = var.multi_az_enabled

  # Network configuration
  subnet_group_name       = aws_elasticache_subnet_group.main.name
  security_group_ids      = [var.security_group_id]

  # Encryption
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  kms_key_id                = aws_kms_key.elasticache.arn
  auth_token                = var.auth_token_enabled ? random_password.auth_token[0].result : null

  # Engine configuration
  engine_version          = var.redis_version
  
  # Backup configuration
  snapshot_retention_limit = var.snapshot_retention_limit
  snapshot_window         = var.snapshot_window
  maintenance_window      = var.maintenance_window

  # Logging
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_slow.name
    destination_type = "cloudwatch-logs"
    log_format       = "text"
    log_type         = "slow-log"
  }

  tags = merge(var.common_tags, {
    Name = "${var.environment}-redis"
  })

  depends_on = [
    aws_cloudwatch_log_group.redis_slow
  ]
}

# Random auth token for Redis
resource "random_password" "auth_token" {
  count = var.auth_token_enabled ? 1 : 0

  length  = 32
  special = false
}

# Store auth token in AWS Secrets Manager
resource "aws_secretsmanager_secret" "redis_auth_token" {
  count = var.auth_token_enabled ? 1 : 0

  name                    = "${var.environment}-redis-auth-token"
  description             = "Auth token for Redis cluster"
  recovery_window_in_days = 7

  tags = var.common_tags
}

resource "aws_secretsmanager_secret_version" "redis_auth_token" {
  count = var.auth_token_enabled ? 1 : 0

  secret_id     = aws_secretsmanager_secret.redis_auth_token[0].id
  secret_string = random_password.auth_token[0].result
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "redis_slow" {
  name              = "/aws/elasticache/${var.environment}-redis/slow-log"
  retention_in_days = 7

  tags = var.common_tags
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "redis_cpu" {
  alarm_name          = "${var.environment}-redis-cpu-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors redis cpu utilization"
  alarm_actions       = var.alarm_actions

  dimensions = {
    CacheClusterId = aws_elasticache_replication_group.main.id
  }

  tags = var.common_tags
}

resource "aws_cloudwatch_metric_alarm" "redis_memory" {
  alarm_name          = "${var.environment}-redis-memory-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "DatabaseMemoryUsagePercentage"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors redis memory utilization"
  alarm_actions       = var.alarm_actions

  dimensions = {
    CacheClusterId = aws_elasticache_replication_group.main.id
  }

  tags = var.common_tags
}

resource "aws_cloudwatch_metric_alarm" "redis_connections" {
  alarm_name          = "${var.environment}-redis-current-connections"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CurrConnections"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Average"
  threshold           = var.connection_threshold
  alarm_description   = "This metric monitors redis current connections"
  alarm_actions       = var.alarm_actions

  dimensions = {
    CacheClusterId = aws_elasticache_replication_group.main.id
  }

  tags = var.common_tags
}