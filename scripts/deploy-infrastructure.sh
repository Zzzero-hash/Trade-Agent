#!/bin/bash

# Production Infrastructure Deployment Script
set -e

# Configuration
ENVIRONMENT="production"
AWS_REGION="us-east-1"
TERRAFORM_DIR="terraform/environments/${ENVIRONMENT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if AWS CLI is installed and configured
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed. Please install it first."
    fi
    
    # Check if Terraform is installed
    if ! command -v terraform &> /dev/null; then
        error "Terraform is not installed. Please install it first."
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed. Please install it first."
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials are not configured or invalid."
    fi
    
    log "Prerequisites check passed."
}

# Initialize Terraform
init_terraform() {
    log "Initializing Terraform..."
    
    cd "${TERRAFORM_DIR}"
    
    # Check if backend.tf exists
    if [ ! -f "backend.tf" ]; then
        warn "backend.tf not found. Please copy backend.tf.example to backend.tf and configure it."
        return 1
    fi
    
    # Check if terraform.tfvars exists
    if [ ! -f "terraform.tfvars" ]; then
        warn "terraform.tfvars not found. Please copy terraform.tfvars.example to terraform.tfvars and configure it."
        return 1
    fi
    
    terraform init
    
    cd - > /dev/null
    log "Terraform initialized successfully."
}

# Plan Terraform deployment
plan_terraform() {
    log "Planning Terraform deployment..."
    
    cd "${TERRAFORM_DIR}"
    terraform plan -out=tfplan
    cd - > /dev/null
    
    log "Terraform plan completed. Review the plan above."
    read -p "Do you want to proceed with the deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Deployment cancelled."
        exit 0
    fi
}

# Apply Terraform deployment
apply_terraform() {
    log "Applying Terraform deployment..."
    
    cd "${TERRAFORM_DIR}"
    terraform apply tfplan
    cd - > /dev/null
    
    log "Terraform deployment completed successfully."
}

# Configure kubectl
configure_kubectl() {
    log "Configuring kubectl for EKS cluster..."
    
    # Get cluster name from Terraform output
    cd "${TERRAFORM_DIR}"
    CLUSTER_NAME=$(terraform output -raw cluster_name)
    cd - > /dev/null
    
    # Update kubeconfig
    aws eks update-kubeconfig --region "${AWS_REGION}" --name "${CLUSTER_NAME}"
    
    # Test connection
    kubectl get nodes
    
    log "kubectl configured successfully."
}

# Deploy Kubernetes resources
deploy_k8s_resources() {
    log "Deploying Kubernetes resources..."
    
    # Get Terraform outputs
    cd "${TERRAFORM_DIR}"
    VPC_ID=$(terraform output -raw vpc_id)
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    cd - > /dev/null
    
    # Replace placeholders in Kubernetes manifests
    sed -i.bak "s/ACCOUNT_ID/${ACCOUNT_ID}/g" k8s/production/*.yaml
    sed -i.bak "s/VPC_ID/${VPC_ID}/g" k8s/production/*.yaml
    
    # Apply namespaces first
    kubectl apply -f k8s/production/namespace.yaml
    
    # Apply cluster autoscaler
    kubectl apply -f k8s/production/cluster-autoscaler.yaml
    
    # Apply AWS Load Balancer Controller
    kubectl apply -f k8s/production/aws-load-balancer-controller.yaml
    
    # Clean up backup files
    rm -f k8s/production/*.yaml.bak
    
    log "Kubernetes resources deployed successfully."
}

# Create IAM roles for service accounts
create_iam_roles() {
    log "Creating IAM roles for service accounts..."
    
    cd "${TERRAFORM_DIR}"
    CLUSTER_NAME=$(terraform output -raw cluster_name)
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    OIDC_ISSUER=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --query "cluster.identity.oidc.issuer" --output text)
    cd - > /dev/null
    
    # Create cluster autoscaler role
    cat > /tmp/cluster-autoscaler-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/${OIDC_ISSUER#https://}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "${OIDC_ISSUER#https://}:sub": "system:serviceaccount:kube-system:cluster-autoscaler",
          "${OIDC_ISSUER#https://}:aud": "sts.amazonaws.com"
        }
      }
    }
  ]
}
EOF

    aws iam create-role \
        --role-name "${ENVIRONMENT}-cluster-autoscaler-role" \
        --assume-role-policy-document file:///tmp/cluster-autoscaler-trust-policy.json \
        --description "Role for cluster autoscaler" || true

    aws iam attach-role-policy \
        --role-name "${ENVIRONMENT}-cluster-autoscaler-role" \
        --policy-arn arn:aws:iam::aws:policy/AutoScalingFullAccess || true

    # Create AWS Load Balancer Controller role
    cat > /tmp/aws-load-balancer-controller-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/${OIDC_ISSUER#https://}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "${OIDC_ISSUER#https://}:sub": "system:serviceaccount:kube-system:aws-load-balancer-controller",
          "${OIDC_ISSUER#https://}:aud": "sts.amazonaws.com"
        }
      }
    }
  ]
}
EOF

    aws iam create-role \
        --role-name "${ENVIRONMENT}-aws-load-balancer-controller-role" \
        --assume-role-policy-document file:///tmp/aws-load-balancer-controller-trust-policy.json \
        --description "Role for AWS Load Balancer Controller" || true

    # Download and apply AWS Load Balancer Controller policy
    curl -o /tmp/iam_policy.json https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.7.2/docs/install/iam_policy.json

    aws iam create-policy \
        --policy-name "${ENVIRONMENT}-AWSLoadBalancerControllerIAMPolicy" \
        --policy-document file:///tmp/iam_policy.json || true

    aws iam attach-role-policy \
        --role-name "${ENVIRONMENT}-aws-load-balancer-controller-role" \
        --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${ENVIRONMENT}-AWSLoadBalancerControllerIAMPolicy" || true

    # Clean up temporary files
    rm -f /tmp/cluster-autoscaler-trust-policy.json
    rm -f /tmp/aws-load-balancer-controller-trust-policy.json
    rm -f /tmp/iam_policy.json

    log "IAM roles created successfully."
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check EKS cluster status
    cd "${TERRAFORM_DIR}"
    CLUSTER_NAME=$(terraform output -raw cluster_name)
    cd - > /dev/null
    
    CLUSTER_STATUS=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --query "cluster.status" --output text)
    if [ "${CLUSTER_STATUS}" != "ACTIVE" ]; then
        error "EKS cluster is not active. Status: ${CLUSTER_STATUS}"
    fi
    
    # Check RDS instance status
    RDS_ENDPOINT=$(cd "${TERRAFORM_DIR}" && terraform output -raw rds_endpoint)
    if [ -z "${RDS_ENDPOINT}" ]; then
        error "RDS endpoint not found in Terraform outputs"
    fi
    
    # Check ElastiCache cluster status
    REDIS_ENDPOINT=$(cd "${TERRAFORM_DIR}" && terraform output -raw redis_primary_endpoint)
    if [ -z "${REDIS_ENDPOINT}" ]; then
        error "Redis endpoint not found in Terraform outputs"
    fi
    
    # Check Kubernetes nodes
    NODE_COUNT=$(kubectl get nodes --no-headers | wc -l)
    if [ "${NODE_COUNT}" -lt 2 ]; then
        error "Insufficient number of Kubernetes nodes. Expected at least 2, got ${NODE_COUNT}"
    fi
    
    # Check system pods
    kubectl get pods -n kube-system
    kubectl get pods -n trading-platform
    
    log "Deployment verification completed successfully."
}

# Main deployment function
main() {
    log "Starting production infrastructure deployment..."
    
    check_prerequisites
    init_terraform
    plan_terraform
    apply_terraform
    configure_kubectl
    create_iam_roles
    deploy_k8s_resources
    verify_deployment
    
    log "Production infrastructure deployment completed successfully!"
    log "Next steps:"
    log "1. Deploy the trading platform application"
    log "2. Configure monitoring and alerting"
    log "3. Set up CI/CD pipelines"
    log "4. Configure backup and disaster recovery procedures"
}

# Run main function
main "$@"