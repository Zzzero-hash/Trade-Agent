#!/bin/bash

# Infrastructure Validation Script
set -e

# Configuration
ENVIRONMENT="production"
AWS_REGION="us-east-1"
TERRAFORM_DIR="terraform/environments/${ENVIRONMENT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠ WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗ ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] ℹ INFO: $1${NC}"
}

# Validation counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Function to run a check
run_check() {
    local check_name="$1"
    local check_command="$2"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    info "Running check: $check_name"
    
    if eval "$check_command" > /dev/null 2>&1; then
        log "$check_name - PASSED"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        error "$check_name - FAILED"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi
}

# Get Terraform outputs
get_terraform_outputs() {
    info "Retrieving Terraform outputs..."
    
    cd "${TERRAFORM_DIR}"
    
    # Check if Terraform state exists
    if ! terraform show > /dev/null 2>&1; then
        error "Terraform state not found. Please deploy infrastructure first."
        exit 1
    fi
    
    # Export outputs as environment variables
    export CLUSTER_NAME=$(terraform output -raw cluster_name 2>/dev/null || echo "")
    export VPC_ID=$(terraform output -raw vpc_id 2>/dev/null || echo "")
    export RDS_ENDPOINT=$(terraform output -raw rds_endpoint 2>/dev/null || echo "")
    export REDIS_ENDPOINT=$(terraform output -raw redis_primary_endpoint 2>/dev/null || echo "")
    export DATA_BUCKET=$(terraform output -raw data_storage_bucket_name 2>/dev/null || echo "")
    
    cd - > /dev/null
    
    log "Terraform outputs retrieved successfully"
}

# Validate AWS CLI and credentials
validate_aws_cli() {
    info "Validating AWS CLI and credentials..."
    
    run_check "AWS CLI installed" "command -v aws"
    run_check "AWS credentials configured" "aws sts get-caller-identity"
    run_check "AWS region set correctly" "[ \"\$(aws configure get region)\" = \"${AWS_REGION}\" ]"
}

# Validate VPC and networking
validate_vpc() {
    info "Validating VPC and networking..."
    
    if [ -z "$VPC_ID" ]; then
        error "VPC ID not found in Terraform outputs"
        return 1
    fi
    
    run_check "VPC exists" "aws ec2 describe-vpcs --vpc-ids $VPC_ID"
    run_check "VPC has DNS support" "aws ec2 describe-vpc-attribute --vpc-id $VPC_ID --attribute enableDnsSupport --query 'EnableDnsSupport.Value'"
    run_check "VPC has DNS hostnames" "aws ec2 describe-vpc-attribute --vpc-id $VPC_ID --attribute enableDnsHostnames --query 'EnableDnsHostnames.Value'"
    
    # Check subnets
    SUBNET_COUNT=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'length(Subnets)' --output text)
    run_check "Sufficient subnets (expected 9, got $SUBNET_COUNT)" "[ $SUBNET_COUNT -eq 9 ]"
    
    # Check NAT gateways
    NAT_COUNT=$(aws ec2 describe-nat-gateways --filter "Name=vpc-id,Values=$VPC_ID" --query 'length(NatGateways[?State==`available`])' --output text)
    run_check "NAT gateways available (expected 3, got $NAT_COUNT)" "[ $NAT_COUNT -eq 3 ]"
}

# Validate EKS cluster
validate_eks() {
    info "Validating EKS cluster..."
    
    if [ -z "$CLUSTER_NAME" ]; then
        error "Cluster name not found in Terraform outputs"
        return 1
    fi
    
    run_check "EKS cluster exists" "aws eks describe-cluster --name $CLUSTER_NAME"
    run_check "EKS cluster is active" "[ \"\$(aws eks describe-cluster --name $CLUSTER_NAME --query 'cluster.status' --output text)\" = \"ACTIVE\" ]"
    
    # Check node groups
    NODE_GROUPS=$(aws eks list-nodegroups --cluster-name $CLUSTER_NAME --query 'nodegroups' --output text)
    NODE_GROUP_COUNT=$(echo $NODE_GROUPS | wc -w)
    run_check "Node groups exist (expected 2-3, got $NODE_GROUP_COUNT)" "[ $NODE_GROUP_COUNT -ge 2 ]"
    
    # Check kubectl connectivity
    run_check "kubectl configured" "kubectl cluster-info"
    run_check "kubectl can list nodes" "kubectl get nodes"
    
    # Check node readiness
    READY_NODES=$(kubectl get nodes --no-headers | grep -c " Ready ")
    TOTAL_NODES=$(kubectl get nodes --no-headers | wc -l)
    run_check "All nodes ready ($READY_NODES/$TOTAL_NODES)" "[ $READY_NODES -eq $TOTAL_NODES ]"
    
    # Check system pods
    run_check "CoreDNS pods running" "kubectl get pods -n kube-system -l k8s-app=kube-dns --field-selector=status.phase=Running"
    run_check "VPC CNI pods running" "kubectl get pods -n kube-system -l k8s-app=aws-node --field-selector=status.phase=Running"
}

# Validate RDS
validate_rds() {
    info "Validating RDS PostgreSQL..."
    
    if [ -z "$RDS_ENDPOINT" ]; then
        error "RDS endpoint not found in Terraform outputs"
        return 1
    fi
    
    DB_INSTANCE_ID="${ENVIRONMENT}-postgres"
    
    run_check "RDS instance exists" "aws rds describe-db-instances --db-instance-identifier $DB_INSTANCE_ID"
    run_check "RDS instance is available" "[ \"\$(aws rds describe-db-instances --db-instance-identifier $DB_INSTANCE_ID --query 'DBInstances[0].DBInstanceStatus' --output text)\" = \"available\" ]"
    run_check "RDS Multi-AZ enabled" "[ \"\$(aws rds describe-db-instances --db-instance-identifier $DB_INSTANCE_ID --query 'DBInstances[0].MultiAZ' --output text)\" = \"True\" ]"
    run_check "RDS encryption enabled" "[ \"\$(aws rds describe-db-instances --db-instance-identifier $DB_INSTANCE_ID --query 'DBInstances[0].StorageEncrypted' --output text)\" = \"True\" ]"
    
    # Check read replica if it exists
    READ_REPLICA_ID="${ENVIRONMENT}-postgres-read-replica"
    if aws rds describe-db-instances --db-instance-identifier $READ_REPLICA_ID > /dev/null 2>&1; then
        run_check "Read replica is available" "[ \"\$(aws rds describe-db-instances --db-instance-identifier $READ_REPLICA_ID --query 'DBInstances[0].DBInstanceStatus' --output text)\" = \"available\" ]"
    fi
}

# Validate ElastiCache
validate_elasticache() {
    info "Validating ElastiCache Redis..."
    
    if [ -z "$REDIS_ENDPOINT" ]; then
        error "Redis endpoint not found in Terraform outputs"
        return 1
    fi
    
    REPLICATION_GROUP_ID="${ENVIRONMENT}-redis"
    
    run_check "Redis replication group exists" "aws elasticache describe-replication-groups --replication-group-id $REPLICATION_GROUP_ID"
    run_check "Redis cluster is available" "[ \"\$(aws elasticache describe-replication-groups --replication-group-id $REPLICATION_GROUP_ID --query 'ReplicationGroups[0].Status' --output text)\" = \"available\" ]"
    run_check "Redis Multi-AZ enabled" "[ \"\$(aws elasticache describe-replication-groups --replication-group-id $REPLICATION_GROUP_ID --query 'ReplicationGroups[0].MultiAZ' --output text)\" = \"enabled\" ]"
    run_check "Redis encryption at rest enabled" "[ \"\$(aws elasticache describe-replication-groups --replication-group-id $REPLICATION_GROUP_ID --query 'ReplicationGroups[0].AtRestEncryptionEnabled' --output text)\" = \"True\" ]"
    run_check "Redis encryption in transit enabled" "[ \"\$(aws elasticache describe-replication-groups --replication-group-id $REPLICATION_GROUP_ID --query 'ReplicationGroups[0].TransitEncryptionEnabled' --output text)\" = \"True\" ]"
}

# Validate S3 buckets
validate_s3() {
    info "Validating S3 buckets..."
    
    if [ -z "$DATA_BUCKET" ]; then
        error "Data bucket name not found in Terraform outputs"
        return 1
    fi
    
    # Get all bucket names from Terraform
    cd "${TERRAFORM_DIR}"
    MODEL_BUCKET=$(terraform output -raw model_artifacts_bucket_name 2>/dev/null || echo "")
    BACKUP_BUCKET=$(terraform output -raw backups_bucket_name 2>/dev/null || echo "")
    COMPLIANCE_BUCKET=$(terraform output -raw compliance_bucket_name 2>/dev/null || echo "")
    cd - > /dev/null
    
    # Validate each bucket
    for bucket in "$DATA_BUCKET" "$MODEL_BUCKET" "$BACKUP_BUCKET" "$COMPLIANCE_BUCKET"; do
        if [ -n "$bucket" ]; then
            run_check "S3 bucket $bucket exists" "aws s3api head-bucket --bucket $bucket"
            run_check "S3 bucket $bucket versioning enabled" "[ \"\$(aws s3api get-bucket-versioning --bucket $bucket --query 'Status' --output text)\" = \"Enabled\" ]"
            run_check "S3 bucket $bucket encryption enabled" "aws s3api get-bucket-encryption --bucket $bucket"
            run_check "S3 bucket $bucket public access blocked" "aws s3api get-public-access-block --bucket $bucket"
        fi
    done
}

# Validate security groups
validate_security_groups() {
    info "Validating security groups..."
    
    if [ -z "$VPC_ID" ]; then
        error "VPC ID not found for security group validation"
        return 1
    fi
    
    # Get security group IDs from Terraform
    cd "${TERRAFORM_DIR}"
    ALB_SG=$(terraform output -raw alb_security_group_id 2>/dev/null || echo "")
    EKS_NODES_SG=$(terraform output -raw eks_nodes_security_group_id 2>/dev/null || echo "")
    cd - > /dev/null
    
    # Validate security groups exist
    if [ -n "$ALB_SG" ]; then
        run_check "ALB security group exists" "aws ec2 describe-security-groups --group-ids $ALB_SG"
    fi
    
    if [ -n "$EKS_NODES_SG" ]; then
        run_check "EKS nodes security group exists" "aws ec2 describe-security-groups --group-ids $EKS_NODES_SG"
    fi
    
    # Count security groups in VPC
    SG_COUNT=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC_ID" --query 'length(SecurityGroups)' --output text)
    run_check "Security groups created (expected 5+, got $SG_COUNT)" "[ $SG_COUNT -ge 5 ]"
}

# Validate IAM roles
validate_iam_roles() {
    info "Validating IAM roles..."
    
    # EKS cluster role
    run_check "EKS cluster role exists" "aws iam get-role --role-name ${ENVIRONMENT}-eks-cluster-role"
    
    # EKS node group role
    run_check "EKS nodes role exists" "aws iam get-role --role-name ${ENVIRONMENT}-eks-nodes-role"
    
    # Service account roles (may not exist if not created yet)
    if aws iam get-role --role-name ${ENVIRONMENT}-cluster-autoscaler-role > /dev/null 2>&1; then
        log "Cluster autoscaler role exists"
    else
        warn "Cluster autoscaler role not found (may need to be created)"
    fi
    
    if aws iam get-role --role-name ${ENVIRONMENT}-aws-load-balancer-controller-role > /dev/null 2>&1; then
        log "AWS Load Balancer Controller role exists"
    else
        warn "AWS Load Balancer Controller role not found (may need to be created)"
    fi
}

# Validate Kubernetes resources
validate_k8s_resources() {
    info "Validating Kubernetes resources..."
    
    # Check namespaces
    run_check "trading-platform namespace exists" "kubectl get namespace trading-platform"
    run_check "monitoring namespace exists" "kubectl get namespace monitoring"
    
    # Check system components
    if kubectl get deployment cluster-autoscaler -n kube-system > /dev/null 2>&1; then
        run_check "Cluster autoscaler deployed" "kubectl get deployment cluster-autoscaler -n kube-system"
        run_check "Cluster autoscaler ready" "[ \"\$(kubectl get deployment cluster-autoscaler -n kube-system -o jsonpath='{.status.readyReplicas}')\" -gt 0 ]"
    else
        warn "Cluster autoscaler not deployed"
    fi
    
    if kubectl get deployment aws-load-balancer-controller -n kube-system > /dev/null 2>&1; then
        run_check "AWS Load Balancer Controller deployed" "kubectl get deployment aws-load-balancer-controller -n kube-system"
        run_check "AWS Load Balancer Controller ready" "[ \"\$(kubectl get deployment aws-load-balancer-controller -n kube-system -o jsonpath='{.status.readyReplicas}')\" -gt 0 ]"
    else
        warn "AWS Load Balancer Controller not deployed"
    fi
}

# Performance and connectivity tests
validate_connectivity() {
    info "Validating connectivity and performance..."
    
    # Test DNS resolution within cluster
    run_check "DNS resolution works" "kubectl run test-dns --image=busybox --rm -it --restart=Never -- nslookup kubernetes.default"
    
    # Test internet connectivity from pods
    run_check "Internet connectivity from pods" "kubectl run test-internet --image=busybox --rm -it --restart=Never -- wget -q --spider http://www.google.com"
    
    # Clean up test pods
    kubectl delete pod test-dns --ignore-not-found=true > /dev/null 2>&1
    kubectl delete pod test-internet --ignore-not-found=true > /dev/null 2>&1
}

# Generate validation report
generate_report() {
    info "Generating validation report..."
    
    echo ""
    echo "=========================================="
    echo "Infrastructure Validation Report"
    echo "=========================================="
    echo "Environment: $ENVIRONMENT"
    echo "Region: $AWS_REGION"
    echo "Timestamp: $(date)"
    echo ""
    echo "Summary:"
    echo "  Total Checks: $TOTAL_CHECKS"
    echo "  Passed: $PASSED_CHECKS"
    echo "  Failed: $FAILED_CHECKS"
    echo ""
    
    if [ $FAILED_CHECKS -eq 0 ]; then
        log "All validation checks passed! Infrastructure is ready for production use."
        echo ""
        echo "Next Steps:"
        echo "1. Deploy the trading platform application"
        echo "2. Configure monitoring and alerting"
        echo "3. Set up CI/CD pipelines"
        echo "4. Perform load testing"
        echo "5. Configure backup verification"
        return 0
    else
        error "$FAILED_CHECKS validation checks failed. Please review and fix the issues."
        echo ""
        echo "Common Solutions:"
        echo "1. Check AWS permissions and credentials"
        echo "2. Verify Terraform deployment completed successfully"
        echo "3. Ensure kubectl is configured correctly"
        echo "4. Check AWS service limits and quotas"
        echo "5. Review CloudWatch logs for error details"
        return 1
    fi
}

# Main validation function
main() {
    info "Starting infrastructure validation for $ENVIRONMENT environment..."
    
    get_terraform_outputs
    validate_aws_cli
    validate_vpc
    validate_eks
    validate_rds
    validate_elasticache
    validate_s3
    validate_security_groups
    validate_iam_roles
    validate_k8s_resources
    validate_connectivity
    
    generate_report
}

# Run main function
main "$@"