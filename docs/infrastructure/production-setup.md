# Production Infrastructure Setup Guide

This guide walks through setting up the production-grade infrastructure for the AI Trading Platform using Terraform and Kubernetes.

## Prerequisites

### Required Tools
- AWS CLI v2.x
- Terraform >= 1.0
- kubectl >= 1.28
- Helm >= 3.x

### AWS Permissions
The deployment requires the following AWS permissions:
- EC2 (VPC, Subnets, Security Groups, Key Pairs)
- EKS (Cluster, Node Groups, Add-ons)
- RDS (PostgreSQL instances, Parameter Groups, Subnet Groups)
- ElastiCache (Redis clusters, Parameter Groups, Subnet Groups)
- S3 (Buckets, Policies, Lifecycle configurations)
- IAM (Roles, Policies, Service Accounts)
- KMS (Keys, Aliases)
- CloudWatch (Log Groups, Alarms)
- Secrets Manager (Secrets, Versions)

## Infrastructure Components

### 1. VPC and Networking
- **VPC**: 10.0.0.0/16 CIDR block
- **Public Subnets**: 3 subnets across AZs for load balancers
- **Private Subnets**: 3 subnets across AZs for application workloads
- **Database Subnets**: 3 subnets across AZs for RDS and ElastiCache
- **NAT Gateways**: High availability with one per AZ
- **Internet Gateway**: For public subnet internet access

### 2. EKS Cluster
- **Kubernetes Version**: 1.28
- **Node Groups**:
  - General: m5.xlarge instances for general workloads
  - ML Workload: c5.4xlarge instances for machine learning tasks
  - Trading Engine: c5.2xlarge instances for trading operations
- **Add-ons**: VPC CNI, CoreDNS, kube-proxy
- **Auto-scaling**: Cluster Autoscaler for dynamic scaling

### 3. RDS PostgreSQL
- **Engine**: PostgreSQL 15.4
- **Instance Class**: db.r6g.2xlarge (production)
- **Multi-AZ**: Enabled for high availability
- **Read Replica**: Enabled for read scaling
- **Backup**: 30-day retention period
- **Encryption**: At rest and in transit
- **Monitoring**: Enhanced monitoring and Performance Insights

### 4. ElastiCache Redis
- **Engine**: Redis 7.0
- **Node Type**: cache.r7g.2xlarge (production)
- **Cluster Mode**: Enabled with 3 nodes
- **Multi-AZ**: Enabled for failover
- **Auth Token**: Enabled for security
- **Encryption**: At rest and in transit
- **Backup**: 7-day snapshot retention

### 5. S3 Storage
- **Data Storage**: Market data and historical information
- **Model Artifacts**: ML model files and checkpoints
- **Backups**: Database and application backups
- **Compliance**: Regulatory documents and audit logs
- **Lifecycle Policies**: Automatic transition to cheaper storage classes
- **Cross-Region Replication**: For compliance bucket (optional)

## Deployment Steps

### 1. Prepare Configuration Files

```bash
# Navigate to production environment
cd terraform/environments/production

# Copy and customize backend configuration
cp backend.tf.example backend.tf
# Edit backend.tf with your S3 bucket and DynamoDB table

# Copy and customize variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your specific values
```

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Plan and Apply Infrastructure

```bash
# Review the deployment plan
terraform plan

# Apply the infrastructure
terraform apply
```

### 4. Configure kubectl

```bash
# Update kubeconfig for the new cluster
aws eks update-kubeconfig --region us-east-1 --name production-eks-cluster

# Verify cluster access
kubectl get nodes
```

### 5. Deploy Kubernetes Components

```bash
# Create namespaces
kubectl apply -f k8s/production/namespace.yaml

# Deploy cluster autoscaler (update ACCOUNT_ID first)
kubectl apply -f k8s/production/cluster-autoscaler.yaml

# Deploy AWS Load Balancer Controller (update ACCOUNT_ID and VPC_ID first)
kubectl apply -f k8s/production/aws-load-balancer-controller.yaml
```

### 6. Create IAM Roles for Service Accounts

The deployment script handles this automatically, but you can also create them manually:

```bash
# Get cluster OIDC issuer
CLUSTER_NAME="production-eks-cluster"
OIDC_ISSUER=$(aws eks describe-cluster --name $CLUSTER_NAME --query "cluster.identity.oidc.issuer" --output text)

# Create roles for cluster autoscaler and AWS Load Balancer Controller
# See scripts/deploy-infrastructure.sh for detailed commands
```

## Configuration Details

### Environment Variables

Key configuration variables in `terraform.tfvars`:

```hcl
# Basic Configuration
aws_region  = "us-east-1"
owner       = "DevOps Team"
cost_center = "Engineering"

# EKS Configuration
kubernetes_version = "1.28"
cluster_endpoint_public_access_cidrs = ["10.0.0.0/8"]  # Restrict access

# RDS Configuration
rds_instance_class = "db.r6g.2xlarge"
rds_multi_az = true
rds_backup_retention_period = 30

# ElastiCache Configuration
elasticache_node_type = "cache.r7g.2xlarge"
elasticache_num_cache_clusters = 3
```

### Security Groups

The infrastructure creates the following security groups:
- **ALB Security Group**: Allows HTTP/HTTPS from internet
- **EKS Cluster Security Group**: Allows HTTPS from VPC
- **EKS Nodes Security Group**: Allows inter-node communication and ALB traffic
- **RDS Security Group**: Allows PostgreSQL from EKS nodes only
- **ElastiCache Security Group**: Allows Redis from EKS nodes only

### Encryption

All data is encrypted:
- **RDS**: Encrypted at rest with KMS, TLS in transit
- **ElastiCache**: Encrypted at rest and in transit with auth token
- **S3**: Server-side encryption with KMS
- **EKS**: Secrets encrypted with KMS

## Monitoring and Observability

### CloudWatch Integration
- **EKS Logs**: API server, audit, authenticator, controller manager, scheduler
- **RDS Logs**: PostgreSQL logs and slow query logs
- **ElastiCache Logs**: Slow log monitoring
- **Application Logs**: Centralized logging via Fluent Bit

### Metrics and Alarms
- **RDS**: CPU, memory, connections, read/write latency
- **ElastiCache**: CPU, memory, connections, cache hit ratio
- **EKS**: Node resource utilization, pod status
- **Custom Metrics**: Trading performance, API latency

## Backup and Disaster Recovery

### Automated Backups
- **RDS**: 30-day automated backups with point-in-time recovery
- **ElastiCache**: Daily snapshots with 7-day retention
- **S3**: Versioning enabled with lifecycle policies
- **EKS**: Velero for cluster backup (to be configured)

### Cross-Region Replication
- **Compliance Data**: Replicated to secondary region
- **Critical Backups**: Cross-region backup strategy
- **Disaster Recovery**: RTO < 4 hours, RPO < 15 minutes

## Cost Optimization

### Resource Sizing
- **Production**: Right-sized for expected load with auto-scaling
- **Development**: Smaller instances with similar architecture
- **Staging**: Medium-sized for testing and validation

### Cost Controls
- **Reserved Instances**: For predictable workloads
- **Spot Instances**: For non-critical batch processing
- **S3 Lifecycle**: Automatic transition to cheaper storage
- **Resource Tagging**: For cost allocation and tracking

## Security Best Practices

### Network Security
- **Private Subnets**: Application workloads isolated from internet
- **Security Groups**: Least privilege access rules
- **NACLs**: Additional network-level protection
- **VPC Flow Logs**: Network traffic monitoring

### Identity and Access
- **IAM Roles**: Service accounts with minimal permissions
- **RBAC**: Kubernetes role-based access control
- **MFA**: Multi-factor authentication for admin access
- **Audit Logging**: All API calls logged and monitored

### Data Protection
- **Encryption**: All data encrypted at rest and in transit
- **Secrets Management**: AWS Secrets Manager for sensitive data
- **Key Rotation**: Automatic rotation of encryption keys
- **Access Logging**: All data access logged and monitored

## Troubleshooting

### Common Issues

1. **EKS Node Group Creation Fails**
   - Check IAM permissions for node group role
   - Verify subnet configuration and availability zones
   - Ensure security groups allow required traffic

2. **RDS Connection Issues**
   - Verify security group rules allow PostgreSQL traffic
   - Check subnet group configuration
   - Validate VPC and DNS resolution

3. **ElastiCache Connection Issues**
   - Verify auth token configuration
   - Check security group rules for Redis traffic
   - Validate subnet group and VPC configuration

4. **Terraform State Issues**
   - Ensure S3 backend bucket exists and is accessible
   - Check DynamoDB table for state locking
   - Verify AWS credentials and permissions

### Useful Commands

```bash
# Check EKS cluster status
aws eks describe-cluster --name production-eks-cluster

# View RDS instance details
aws rds describe-db-instances --db-instance-identifier production-postgres

# Check ElastiCache cluster status
aws elasticache describe-replication-groups --replication-group-id production-redis

# View Terraform state
terraform show

# Get all outputs
terraform output
```

## Next Steps

After infrastructure deployment:

1. **Application Deployment**: Deploy the trading platform services
2. **Monitoring Setup**: Configure Prometheus, Grafana, and alerting
3. **CI/CD Pipeline**: Set up automated deployment pipelines
4. **Security Hardening**: Implement additional security measures
5. **Performance Testing**: Validate system performance under load
6. **Disaster Recovery Testing**: Test backup and recovery procedures

## Support and Maintenance

### Regular Tasks
- **Security Updates**: Keep all components updated
- **Backup Verification**: Regular restore testing
- **Performance Monitoring**: Continuous optimization
- **Cost Review**: Monthly cost analysis and optimization
- **Capacity Planning**: Monitor growth and scale accordingly

### Emergency Procedures
- **Incident Response**: Defined escalation procedures
- **Disaster Recovery**: Step-by-step recovery process
- **Security Incidents**: Immediate response protocols
- **Performance Issues**: Troubleshooting and resolution steps