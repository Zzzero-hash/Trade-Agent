#!/bin/bash

# AI Trading Platform Cluster Setup Script
# Requirements: 6.1, 6.3, 6.6

set -euo pipefail

# Configuration
CLUSTER_NAME="${CLUSTER_NAME:-ai-trading-cluster}"
REGION="${REGION:-us-west-2}"
NODE_TYPE="${NODE_TYPE:-m5.large}"
MIN_NODES="${MIN_NODES:-3}"
MAX_NODES="${MAX_NODES:-10}"
KUBERNETES_VERSION="${KUBERNETES_VERSION:-1.28}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required tools
    local required_tools=("aws" "kubectl" "helm" "eksctl")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create EKS cluster
create_eks_cluster() {
    log_info "Creating EKS cluster: $CLUSTER_NAME"
    
    # Check if cluster already exists
    if aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" &> /dev/null; then
        log_warning "Cluster $CLUSTER_NAME already exists"
        return 0
    fi
    
    # Create cluster configuration
    cat > cluster-config.yaml << EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: $CLUSTER_NAME
  region: $REGION
  version: "$KUBERNETES_VERSION"

nodeGroups:
  - name: general-workers
    instanceType: $NODE_TYPE
    minSize: $MIN_NODES
    maxSize: $MAX_NODES
    desiredCapacity: $MIN_NODES
    volumeSize: 100
    volumeType: gp3
    ssh:
      allow: false
    iam:
      withAddonPolicies:
        autoScaler: true
        cloudWatch: true
        ebs: true
        efs: true
        albIngress: true
    labels:
      workload-type: general
    tags:
      Environment: production
      Project: ai-trading-platform

  - name: ml-workers
    instanceType: g4dn.xlarge
    minSize: 0
    maxSize: 5
    desiredCapacity: 0
    volumeSize: 200
    volumeType: gp3
    ssh:
      allow: false
    iam:
      withAddonPolicies:
        autoScaler: true
        cloudWatch: true
        ebs: true
    labels:
      workload-type: ml
      nvidia.com/gpu: "true"
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
    tags:
      Environment: production
      Project: ai-trading-platform

addons:
  - name: vpc-cni
    version: latest
  - name: coredns
    version: latest
  - name: kube-proxy
    version: latest
  - name: aws-ebs-csi-driver
    version: latest

cloudWatch:
  clusterLogging:
    enable: ["api", "audit", "authenticator", "controllerManager", "scheduler"]
EOF

    # Create the cluster
    eksctl create cluster -f cluster-config.yaml
    
    if [ $? -eq 0 ]; then
        log_success "EKS cluster created successfully"
    else
        log_error "Failed to create EKS cluster"
        exit 1
    fi
    
    # Clean up config file
    rm -f cluster-config.yaml
}

# Install cluster addons
install_addons() {
    log_info "Installing cluster addons..."
    
    # Add Helm repositories
    helm repo add aws-ebs-csi-driver https://kubernetes-sigs.github.io/aws-ebs-csi-driver
    helm repo add aws-load-balancer-controller https://aws.github.io/aws-load-balancer-controller
    helm repo add metrics-server https://kubernetes-sigs.github.io/metrics-server
    helm repo add cluster-autoscaler https://kubernetes.github.io/autoscaler
    helm repo add nvidia https://nvidia.github.io/k8s-device-plugin
    helm repo update
    
    # Install AWS Load Balancer Controller
    log_info "Installing AWS Load Balancer Controller..."
    kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"
    
    helm upgrade --install aws-load-balancer-controller aws-load-balancer-controller/aws-load-balancer-controller \
        --namespace kube-system \
        --set clusterName="$CLUSTER_NAME" \
        --set serviceAccount.create=false \
        --set serviceAccount.name=aws-load-balancer-controller
    
    # Install Metrics Server
    log_info "Installing Metrics Server..."
    helm upgrade --install metrics-server metrics-server/metrics-server \
        --namespace kube-system \
        --set args[0]="--kubelet-insecure-tls"
    
    # Install Cluster Autoscaler
    log_info "Installing Cluster Autoscaler..."
    helm upgrade --install cluster-autoscaler cluster-autoscaler/cluster-autoscaler \
        --namespace kube-system \
        --set autoDiscovery.clusterName="$CLUSTER_NAME" \
        --set awsRegion="$REGION"
    
    # Install NVIDIA Device Plugin (for GPU nodes)
    log_info "Installing NVIDIA Device Plugin..."
    helm upgrade --install nvidia-device-plugin nvidia/nvidia-device-plugin \
        --namespace kube-system \
        --set nodeSelector."workload-type"="ml"
    
    log_success "Cluster addons installed successfully"
}

# Configure storage classes
configure_storage() {
    log_info "Configuring storage classes..."
    
    # Create fast SSD storage class
    cat > fast-ssd-storageclass.yaml << EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
EOF

    kubectl apply -f fast-ssd-storageclass.yaml
    rm -f fast-ssd-storageclass.yaml
    
    log_success "Storage classes configured"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Add monitoring repositories
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Create monitoring namespace
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
    
    # Install Prometheus
    log_info "Installing Prometheus..."
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=fast-ssd \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
        --set grafana.persistence.enabled=true \
        --set grafana.persistence.storageClassName=fast-ssd \
        --set grafana.persistence.size=10Gi
    
    log_success "Monitoring setup completed"
}

# Configure RBAC
configure_rbac() {
    log_info "Configuring RBAC..."
    
    # Create service account for deployments
    cat > deployment-rbac.yaml << EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: deployment-sa
  namespace: kube-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: deployment-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["networking.k8s.io"]
  resources: ["ingresses"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: deployment-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: deployment-role
subjects:
- kind: ServiceAccount
  name: deployment-sa
  namespace: kube-system
EOF

    kubectl apply -f deployment-rbac.yaml
    rm -f deployment-rbac.yaml
    
    log_success "RBAC configured"
}

# Verify cluster setup
verify_cluster() {
    log_info "Verifying cluster setup..."
    
    # Check cluster info
    kubectl cluster-info
    
    # Check nodes
    log_info "Cluster nodes:"
    kubectl get nodes -o wide
    
    # Check system pods
    log_info "System pods:"
    kubectl get pods -n kube-system
    
    # Check storage classes
    log_info "Storage classes:"
    kubectl get storageclass
    
    # Check if monitoring is working
    log_info "Monitoring pods:"
    kubectl get pods -n monitoring
    
    log_success "Cluster verification completed"
}

# Main function
main() {
    log_info "Setting up AI Trading Platform Kubernetes cluster"
    log_info "Cluster: $CLUSTER_NAME"
    log_info "Region: $REGION"
    log_info "Node Type: $NODE_TYPE"
    log_info "Nodes: $MIN_NODES-$MAX_NODES"
    echo
    
    check_prerequisites
    create_eks_cluster
    install_addons
    configure_storage
    setup_monitoring
    configure_rbac
    verify_cluster
    
    echo
    log_success "Cluster setup completed successfully!"
    
    # Display connection info
    echo
    log_info "Connection Information:"
    echo "  Cluster Name: $CLUSTER_NAME"
    echo "  Region: $REGION"
    echo "  Kubeconfig: ~/.kube/config"
    echo
    log_info "To connect to the cluster:"
    echo "  aws eks update-kubeconfig --region $REGION --name $CLUSTER_NAME"
    echo
    log_info "To access Grafana:"
    echo "  kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80"
    echo "  Default credentials: admin/prom-operator"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster-name)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --node-type)
            NODE_TYPE="$2"
            shift 2
            ;;
        --min-nodes)
            MIN_NODES="$2"
            shift 2
            ;;
        --max-nodes)
            MAX_NODES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --cluster-name NAME    Cluster name (default: ai-trading-cluster)"
            echo "  --region REGION        AWS region (default: us-west-2)"
            echo "  --node-type TYPE       EC2 instance type (default: m5.large)"
            echo "  --min-nodes NUM        Minimum nodes (default: 3)"
            echo "  --max-nodes NUM        Maximum nodes (default: 10)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main