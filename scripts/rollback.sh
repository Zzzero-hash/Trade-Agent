#!/bin/bash

# AI Trading Platform Rollback Script
# Requirements: 6.1, 6.3, 6.6

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${ENVIRONMENT:-staging}"
NAMESPACE="ai-trading-platform"
TIMEOUT="${TIMEOUT:-300}"

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
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# List available rollback targets
list_rollback_targets() {
    log_info "Available rollback targets:"
    
    local deployments=("api-deployment" "ml-worker-deployment" "data-processor-deployment")
    
    for deployment in "${deployments[@]}"; do
        echo
        log_info "Deployment: $deployment"
        
        # Get rollout history
        kubectl rollout history deployment/"$deployment" -n "$NAMESPACE" 2>/dev/null || {
            log_warning "No rollout history found for $deployment"
            continue
        }
    done
}

# Rollback specific deployment
rollback_deployment() {
    local deployment="$1"
    local revision="${2:-}"
    
    log_info "Rolling back $deployment..."
    
    # Check if deployment exists
    if ! kubectl get deployment "$deployment" -n "$NAMESPACE" &> /dev/null; then
        log_error "Deployment $deployment does not exist in namespace $NAMESPACE"
        return 1
    fi
    
    # Perform rollback
    if [ -n "$revision" ]; then
        log_info "Rolling back $deployment to revision $revision"
        kubectl rollout undo deployment/"$deployment" -n "$NAMESPACE" --to-revision="$revision"
    else
        log_info "Rolling back $deployment to previous revision"
        kubectl rollout undo deployment/"$deployment" -n "$NAMESPACE"
    fi
    
    if [ $? -eq 0 ]; then
        log_success "Rollback command executed for $deployment"
    else
        log_error "Failed to execute rollback for $deployment"
        return 1
    fi
    
    # Wait for rollback to complete
    log_info "Waiting for $deployment rollback to complete..."
    kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout="${TIMEOUT}s"
    
    if [ $? -eq 0 ]; then
        log_success "$deployment rollback completed successfully"
    else
        log_error "$deployment rollback failed or timed out"
        return 1
    fi
}

# Rollback all deployments
rollback_all_deployments() {
    local revision="${1:-}"
    
    log_info "Rolling back all deployments..."
    
    local deployments=("api-deployment" "ml-worker-deployment" "data-processor-deployment")
    local failed_deployments=()
    
    for deployment in "${deployments[@]}"; do
        if ! rollback_deployment "$deployment" "$revision"; then
            failed_deployments+=("$deployment")
        fi
    done
    
    if [ ${#failed_deployments[@]} -eq 0 ]; then
        log_success "All deployments rolled back successfully"
    else
        log_error "Failed to rollback deployments: ${failed_deployments[*]}"
        exit 1
    fi
}

# Verify rollback
verify_rollback() {
    log_info "Verifying rollback..."
    
    # Wait a bit for services to stabilize
    sleep 30
    
    # Check pod status
    log_info "Checking pod status..."
    kubectl get pods -n "$NAMESPACE"
    
    # Check if all pods are running
    local not_running_pods
    not_running_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
    
    if [ "$not_running_pods" -gt 0 ]; then
        log_warning "Some pods are not in Running state"
        kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running
    else
        log_success "All pods are running"
    fi
    
    # Run health checks
    run_health_checks
}

# Run health checks
run_health_checks() {
    log_info "Running post-rollback health checks..."
    
    # Get service endpoints
    local api_url
    if kubectl get ingress trading-platform-ingress -n "$NAMESPACE" &> /dev/null; then
        api_url="https://$(kubectl get ingress trading-platform-ingress -n "$NAMESPACE" -o jsonpath='{.spec.rules[0].host}')"
    else
        # Use port-forward for testing
        kubectl port-forward service/api-service 8080:80 -n "$NAMESPACE" &
        local port_forward_pid=$!
        sleep 5
        api_url="http://localhost:8080"
    fi
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    local health_response
    health_response=$(curl -s -f "$api_url/health" || echo "FAILED")
    
    if [ "$health_response" != "FAILED" ]; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        
        # Show recent logs for debugging
        log_info "Recent logs from API pods:"
        kubectl logs -l app=api -n "$NAMESPACE" --tail=50
        
        # Kill port-forward if it was started
        if [ -n "${port_forward_pid:-}" ]; then
            kill $port_forward_pid 2>/dev/null || true
        fi
        
        exit 1
    fi
    
    # Kill port-forward if it was started
    if [ -n "${port_forward_pid:-}" ]; then
        kill $port_forward_pid 2>/dev/null || true
    fi
}

# Emergency rollback (fastest possible)
emergency_rollback() {
    log_warning "Performing EMERGENCY rollback..."
    
    local deployments=("api-deployment" "ml-worker-deployment" "data-processor-deployment")
    
    # Rollback all deployments in parallel
    for deployment in "${deployments[@]}"; do
        (
            log_info "Emergency rollback of $deployment"
            kubectl rollout undo deployment/"$deployment" -n "$NAMESPACE"
        ) &
    done
    
    # Wait for all background jobs to complete
    wait
    
    # Wait for rollouts to complete
    for deployment in "${deployments[@]}"; do
        log_info "Waiting for $deployment emergency rollback..."
        kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout="60s" || {
            log_error "Emergency rollback of $deployment timed out"
        }
    done
    
    log_warning "Emergency rollback completed. Please verify system status."
}

# Show current deployment status
show_status() {
    log_info "Current deployment status:"
    
    echo
    log_info "Deployments:"
    kubectl get deployments -n "$NAMESPACE"
    
    echo
    log_info "Pods:"
    kubectl get pods -n "$NAMESPACE"
    
    echo
    log_info "Services:"
    kubectl get services -n "$NAMESPACE"
    
    echo
    log_info "Recent events:"
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
}

# Cleanup function
cleanup() {
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
}

# Main function
main() {
    local action="${1:-}"
    local target="${2:-}"
    local revision="${3:-}"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    case "$action" in
        "list"|"ls")
            check_prerequisites
            list_rollback_targets
            ;;
        "deployment"|"deploy")
            if [ -z "$target" ]; then
                log_error "Deployment name required for rollback"
                echo "Usage: $0 deployment <deployment-name> [revision]"
                exit 1
            fi
            check_prerequisites
            rollback_deployment "$target" "$revision"
            verify_rollback
            ;;
        "all")
            check_prerequisites
            rollback_all_deployments "$target"  # target becomes revision in this case
            verify_rollback
            ;;
        "emergency")
            check_prerequisites
            emergency_rollback
            ;;
        "status")
            check_prerequisites
            show_status
            ;;
        "verify")
            check_prerequisites
            verify_rollback
            ;;
        *)
            echo "AI Trading Platform Rollback Script"
            echo
            echo "Usage: $0 <action> [options]"
            echo
            echo "Actions:"
            echo "  list                    List available rollback targets"
            echo "  deployment <name> [rev] Rollback specific deployment"
            echo "  all [revision]          Rollback all deployments"
            echo "  emergency               Emergency rollback (fastest)"
            echo "  status                  Show current deployment status"
            echo "  verify                  Verify current deployment health"
            echo
            echo "Examples:"
            echo "  $0 list"
            echo "  $0 deployment api-deployment"
            echo "  $0 deployment api-deployment 3"
            echo "  $0 all"
            echo "  $0 all 2"
            echo "  $0 emergency"
            echo
            echo "Environment Variables:"
            echo "  ENVIRONMENT    Deployment environment (default: staging)"
            echo "  NAMESPACE      Kubernetes namespace (default: ai-trading-platform)"
            echo "  TIMEOUT        Rollback timeout in seconds (default: 300)"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"