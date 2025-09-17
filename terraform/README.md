# AI Trading Platform Infrastructure

This directory contains Terraform modules and configurations for deploying the production-grade infrastructure for the AI Trading Platform.

## Architecture Overview

The infrastructure is designed following AWS Well-Architected principles with emphasis on:
- **Reliability**: Multi-AZ deployment with automated failover
- **Security**: Encryption at rest and in transit, network isolation
- **Performance**: Auto-scaling and optimized instance types
- **Cost Optimization**: Right-sized resources with lifecycle policies
- **Operational Excellence**: Infrastructure as Code with monitoring

## Directory Structure

```
terraform/
├── modules/                    # Reusable Terraform modules
│   ├── vpc/                   # VPC and networking components
│   ├── security-groups/       # Security group definitions
│   ├── eks/                   # EKS cluster and node groups
│   ├── rds/                   # PostgreSQL database setup
│   ├── elasticache/           # Redis cluster configuration
│   └── s3/                    # S3 buckets and policies
├── environments/              # Environment-specific configurations
│   └── production/            # Production environment
│       ├── main.tf           # Main configuration
│       ├── variables.tf      # Variable definitions
│       ├── outputs.tf        # Output values
│       ├── terraform.tfvars.example  # Example variables
│       └── backend.tf.example        # Example backend config
└── README.md                  # This file
```

## Quick Start

### Prerequisites

1. **AWS CLI** configured with appropriate permissions
2. **Terraform** >= 1.0 installed
3. **kubectl** for Kubernetes management
4. **S3 bucket** for Terraform state storage
5. **DynamoDB table** for state locking

### Deployment Steps

1. **Configure Backend**
   ```bash
   cd terraform/environments/production
   cp backend.tf.example backend.tf
   # Edit backend.tf with your S3 bucket details
   ```

2. **Set Variables**
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your configuration
   ```

3. **Deploy Infrastructure**
   ```bash
   terraform init
   terraform plan
   terraform apply
   ```

4. **Configure kubectl**
   ```bash
   aws eks update-kubeconfig --region us-east-1 --name production-eks-cluster
   ```

5. **Deploy Kubernetes Components**
   ```bash
   kubectl apply -f ../../k8s/production/
   ```

## Module Documentation

### VPC Module (`modules/vpc/`)

Creates a production-ready VPC with:
- Public, private, and database subnets across 3 AZs
- Internet Gateway and NAT Gateways for connectivity
- Route tables with appropriate routing rules
- VPC endpoints for AWS services (optional)

**Key Resources:**
- VPC with DNS support enabled
- 9 subnets (3 public, 3 private, 3 database)
- 3 NAT Gateways for high availability
- Route tables and associations

### Security Groups Module (`modules/security-groups/`)

Defines security groups following least privilege principles:
- ALB security group for load balancer traffic
- EKS cluster and node security groups
- RDS security group for database access
- ElastiCache security group for Redis access

**Security Features:**
- Ingress rules limited to required ports and sources
- Egress rules for necessary outbound traffic
- Inter-service communication properly configured

### EKS Module (`modules/eks/`)

Deploys a production-grade Kubernetes cluster:
- EKS cluster with managed control plane
- Multiple node groups for different workload types
- Cluster add-ons (VPC CNI, CoreDNS, kube-proxy)
- IAM roles and policies for cluster operation

**Key Features:**
- Kubernetes 1.28 with regular updates
- Auto-scaling node groups
- Encryption of secrets at rest
- CloudWatch logging enabled

### RDS Module (`modules/rds/`)

Sets up PostgreSQL database with enterprise features:
- Multi-AZ deployment for high availability
- Read replica for read scaling
- Automated backups with point-in-time recovery
- Enhanced monitoring and Performance Insights

**Key Features:**
- PostgreSQL 15.4 with optimized parameters
- Encryption at rest with KMS
- Automated backup with 30-day retention
- Performance monitoring and alerting

### ElastiCache Module (`modules/elasticache/`)

Configures Redis cluster for caching and sessions:
- Redis 7.0 with cluster mode enabled
- Multi-AZ with automatic failover
- Auth token for security
- Encryption at rest and in transit

**Key Features:**
- 3-node cluster for high availability
- Automated snapshots with retention
- CloudWatch monitoring and alarms
- Parameter group optimization

### S3 Module (`modules/s3/`)

Creates S3 buckets for different data types:
- Data storage bucket for market data
- Model artifacts bucket for ML models
- Backups bucket for database backups
- Compliance bucket for regulatory documents

**Key Features:**
- Versioning enabled on all buckets
- Lifecycle policies for cost optimization
- Server-side encryption with KMS
- Cross-region replication for compliance data

## Environment Configuration

### Production Environment

The production environment is configured for:
- **High Availability**: Multi-AZ deployment across 3 availability zones
- **Scalability**: Auto-scaling groups and cluster autoscaler
- **Security**: Network isolation and encryption everywhere
- **Monitoring**: Comprehensive logging and metrics collection
- **Backup**: Automated backups with cross-region replication

### Key Configuration Parameters

```hcl
# Instance sizing for production workloads
rds_instance_class = "db.r6g.2xlarge"
elasticache_node_type = "cache.r7g.2xlarge"

# High availability settings
rds_multi_az = true
elasticache_multi_az_enabled = true
elasticache_automatic_failover_enabled = true

# Security settings
rds_deletion_protection = true
elasticache_auth_token_enabled = true

# Backup and retention
rds_backup_retention_period = 30
elasticache_snapshot_retention_limit = 7
```

## Monitoring and Alerting

### CloudWatch Integration

All components are integrated with CloudWatch for:
- **Metrics Collection**: CPU, memory, disk, network metrics
- **Log Aggregation**: Application and system logs
- **Alarms**: Automated alerting on threshold breaches
- **Dashboards**: Visual monitoring of system health

### Key Metrics Monitored

- **EKS**: Node resource utilization, pod status, cluster health
- **RDS**: CPU utilization, connections, read/write latency
- **ElastiCache**: CPU utilization, memory usage, cache hit ratio
- **Application**: Custom business metrics and performance indicators

## Security Considerations

### Network Security

- **VPC Isolation**: Private subnets for application workloads
- **Security Groups**: Least privilege access rules
- **Network ACLs**: Additional layer of network security
- **VPC Flow Logs**: Network traffic monitoring and analysis

### Data Protection

- **Encryption at Rest**: All data encrypted using AWS KMS
- **Encryption in Transit**: TLS 1.3 for all communications
- **Key Management**: Automatic key rotation and secure storage
- **Access Control**: IAM roles and policies with minimal permissions

### Compliance Features

- **Audit Logging**: All API calls and data access logged
- **Data Retention**: Configurable retention policies
- **Backup Verification**: Regular backup integrity checks
- **Compliance Reporting**: Automated compliance report generation

## Cost Optimization

### Resource Optimization

- **Right Sizing**: Instances sized for actual workload requirements
- **Auto Scaling**: Dynamic scaling based on demand
- **Reserved Instances**: Cost savings for predictable workloads
- **Spot Instances**: Cost-effective compute for batch processing

### Storage Optimization

- **S3 Lifecycle Policies**: Automatic transition to cheaper storage classes
- **EBS Optimization**: GP3 volumes with optimized IOPS and throughput
- **Backup Optimization**: Intelligent tiering for backup storage
- **Data Compression**: Compression for archived data

## Disaster Recovery

### Backup Strategy

- **RDS**: Automated backups with 30-day retention and point-in-time recovery
- **ElastiCache**: Daily snapshots with 7-day retention
- **S3**: Versioning and cross-region replication for critical data
- **EKS**: Cluster configuration backup and restore procedures

### Recovery Procedures

- **RTO Target**: < 4 hours for full system recovery
- **RPO Target**: < 15 minutes for data loss
- **Automated Failover**: Database and cache automatic failover
- **Manual Procedures**: Documented steps for disaster scenarios

## Troubleshooting

### Common Issues

1. **Terraform State Lock**: Clear DynamoDB lock if deployment fails
2. **IAM Permissions**: Ensure sufficient permissions for all resources
3. **Subnet Capacity**: Verify subnet CIDR blocks don't overlap
4. **Resource Limits**: Check AWS service limits and quotas

### Useful Commands

```bash
# Check Terraform state
terraform state list
terraform show

# Validate configuration
terraform validate
terraform fmt -check

# Import existing resources
terraform import aws_instance.example i-1234567890abcdef0

# Refresh state
terraform refresh
```

## Contributing

### Development Workflow

1. Create feature branch from main
2. Make changes to Terraform modules
3. Test changes in development environment
4. Submit pull request with detailed description
5. Review and approve changes
6. Deploy to staging for validation
7. Deploy to production after approval

### Code Standards

- Use consistent naming conventions
- Add comments for complex configurations
- Follow Terraform best practices
- Include variable descriptions and types
- Add outputs for important resource attributes

### Testing

- Validate Terraform syntax with `terraform validate`
- Format code with `terraform fmt`
- Test in development environment before production
- Use `terraform plan` to review changes
- Implement automated testing where possible

## Support

For infrastructure support and questions:
- Create GitHub issue for bugs or feature requests
- Contact DevOps team for urgent production issues
- Refer to AWS documentation for service-specific questions
- Use Terraform documentation for configuration help

## License

This infrastructure code is proprietary to the AI Trading Platform project.