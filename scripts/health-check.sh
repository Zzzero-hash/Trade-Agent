#!/bin/bash

# AI Trading Platform Health Check Script
# Requirements: 6.1, 6.3, 6.6

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${ENVIRONMENT:-staging}"
NAMESPACE="ai-trading-platform"
TIMEOUT="${TIMEOUT:-30}"

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

# Health check results
declare -A health_results

# Check Kubernetes cluster connectivity
check_cluster_connectivity() {
    log_info "Checking Kubernetes cluster connectivity..."
    
    if kubectl cluster-info &> /dev/null; then
        health_results["cluster"]="healthy"
        log_success "Kubernetes cluster is accessible"
    else
        health_results["cluster"]="unhealthy"
        log_error "Cannot connect to Kubernetes cluster"
        return 1
    fi
}

# Check namespace existence
check_namespace() {
    log_info "Checking namespace $NAMESPACE..."
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        health_results["namespace"]="healthy"
        log_success "Namespace $NAMESPACE exists"
    else
        health_results["namespace"]="unhealthy"
        log_error "Namespace $NAMESPACE does not exist"
        return 1
    fi
}

# Check pod status
check_pods() {
    log_info "Checking pod status..."
    
    local total_pods
    local running_pods
    local pending_pods
    local failed_pods
    
    total_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    running_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
    pending_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Pending --no-headers 2>/dev/null | wc -l)
    failed_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Failed --no-headers 2>/dev/null | wc -l)
    
    if [ "$total_pods" -eq 0 ]; then
        health_results["pods"]="unhealthy"
        log_error "No pods found in namespace $NAMESPACE"
        return 1
    fi
    
    if [ "$failed_pods" -gt 0 ]; then
        health_results["pods"]="degraded"
        log_warning "$failed_pods pods are in Failed state"
        
        # Show failed pods
        log_info "Failed pods:"
        kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Failed
    elif [ "$pending_pods" -gt 0 ]; then
        health_results["pods"]="degraded"
        log_warning "$pending_pods pods are in Pending state"
        
        # Show pending pods
        log_info "Pending pods:"
        kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Pending
    elif [ "$running_pods" -eq "$total_pods" ]; then
        health_results["pods"]="healthy"
        log_success "All $total_pods pods are running"
    else
        health_results["pods"]="degraded"
        log_warning "$running_pods/$total_pods pods are running"
    fi
}

# Check deployment status
check_deployments() {
    log_info "Checking deployment status..."
    
    local deployments=("api-deployment" "ml-worker-deployment" "data-processor-deployment")
    local healthy_deployments=0
    local total_deployments=${#deployments[@]}
    
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "$deployment" -n "$NAMESPACE" &> /dev/null; then
            local ready_replicas
            local desired_replicas
            
            ready_replicas=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
            desired_replicas=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
            
            if [ "$ready_replicas" = "$desired_replicas" ] && [ "$ready_replicas" != "0" ]; then
                log_success "$deployment: $ready_replicas/$desired_replicas replicas ready"
                ((healthy_deployments++))
            else
                log_warning "$deployment: $ready_replicas/$desired_replicas replicas ready"
            fi
        else
            log_error "$deployment: deployment not found"
        fi
    done
    
    if [ "$healthy_deployments" -eq "$total_deployments" ]; then
        health_results["deployments"]="healthy"
        log_success "All deployments are healthy"
    elif [ "$healthy_deployments" -gt 0 ]; then
        health_results["deployments"]="degraded"
        log_warning "$healthy_deployments/$total_deployments deployments are healthy"
    else
        health_results["deployments"]="unhealthy"
        log_error "No deployments are healthy"
    fi
}

# Check service status
check_services() {
    log_info "Checking service status..."
    
    local services=("api-service" "postgres-service" "redis-service")
    local healthy_services=0
    local total_services=${#services[@]}
    
    for service in "${services[@]}"; do
        if kubectl get service "$service" -n "$NAMESPACE" &> /dev/null; then
            local endpoints
            endpoints=$(kubectl get endpoints "$service" -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null || echo "")
            
            if [ -n "$endpoints" ]; then
                log_success "$service: endpoints available"
                ((healthy_services++))
            else
                log_warning "$service: no endpoints available"
            fi
        else
            log_error "$service: service not found"
        fi
    done
    
    if [ "$healthy_services" -eq "$total_services" ]; then
        health_results["services"]="healthy"
        log_success "All services have endpoints"
    elif [ "$healthy_services" -gt 0 ]; then
        health_results["services"]="degraded"
        log_warning "$healthy_services/$total_services services have endpoints"
    else
        health_results["services"]="unhealthy"
        log_error "No services have endpoints"
    fi
}

# Check ingress status
check_ingress() {
    log_info "Checking ingress status..."
    
    if kubectl get ingress trading-platform-ingress -n "$NAMESPACE" &> /dev/null; then
        local ingress_ip
        ingress_ip=$(kubectl get ingress trading-platform-ingress -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
        
        if [ -n "$ingress_ip" ]; then
            health_results["ingress"]="healthy"
            log_success "Ingress has external IP: $ingress_ip"
        else
            health_results["ingress"]="degraded"
            log_warning "Ingress exists but no external IP assigned"
        fi
    else
        health_results["ingress"]="unhealthy"
        log_error "Ingress not found"
    fi
}

# Check API health endpoint
check_api_health() {
    log_info "Checking API health endpoint..."
    
    local api_url=""
    local port_forward_pid=""
    
    # Try to get ingress URL first
    if kubectl get ingress trading-platform-ingress -n "$NAMESPACE" &> /dev/null; then
        local ingress_host
        ingress_host=$(kubectl get ingress trading-platform-ingress -n "$NAMESPACE" -o jsonpath='{.spec.rules[0].host}' 2>/dev/null || echo "")
        
        if [ -n "$ingress_host" ]; then
            api_url="https://$ingress_host"
        fi
    fi
    
    # Fallback to port-forward
    if [ -z "$api_url" ]; then
        log_info "Using port-forward to access API..."
        kubectl port-forward service/api-service 8080:80 -n "$NAMESPACE" &
        port_forward_pid=$!
        sleep 5
        api_url="http://localhost:8080"
    fi
    
    # Test health endpoint
    local health_response
    health_response=$(curl -s -f --max-time "$TIMEOUT" "$api_url/health" 2>/dev/null || echo "FAILED")
    
    # Clean up port-forward
    if [ -n "$port_forward_pid" ]; then
        kill $port_forward_pid 2>/dev/null || true
        wait $port_forward_pid 2>/dev/null || true
    fi
    
    if [ "$health_response" != "FAILED" ]; then
        health_results["api_health"]="healthy"
        log_success "API health endpoint is responding"
        
        # Try to parse response
        if echo "$health_response" | jq . &>/dev/null; then
            local status
            status=$(echo "$health_response" | jq -r '.status // "unknown"')
            log_info "API reports status: $status"
        fi
    else
        health_results["api_health"]="unhealthy"
        log_error "API health endpoint is not responding"
    fi
}

# Check database connectivity
check_database() {
    log_info "Checking database connectivity..."
    
    # Check if postgres pod is running
    local postgres_pod
    postgres_pod=$(kubectl get pods -n "$NAMESPACE" -l app=postgres --no-headers 2>/dev/null | head -1 | awk '{print $1}' || echo "")
    
    if [ -n "$postgres_pod" ]; then
        # Test database connection
        local db_test
        db_test=$(kubectl exec "$postgres_pod" -n "$NAMESPACE" -- pg_isready -U trading_user -d trading_platform 2>/dev/null || echo "FAILED")
        
        if [[ "$db_test" == *"accepting connections"* ]]; then
            health_results["database"]="healthy"
            log_success "Database is accepting connections"
        else
            health_results["database"]="unhealthy"
            log_error "Database is not accepting connections"
        fi
    else
        health_results["database"]="unhealthy"
        log_error "PostgreSQL pod not found"
    fi
}

# Check Redis connectivity
check_redis() {
    log_info "Checking Redis connectivity..."
    
    # Check if redis pod is running
    local redis_pod
    redis_pod=$(kubectl get pods -n "$NAMESPACE" -l app=redis --no-headers 2>/dev/null | head -1 | awk '{print $1}' || echo "")
    
    if [ -n "$redis_pod" ]; then
        # Test Redis connection
        local redis_test
        redis_test=$(kubectl exec "$redis_pod" -n "$NAMESPACE" -- redis-cli ping 2>/dev/null || echo "FAILED")
        
        if [ "$redis_test" = "PONG" ]; then
            health_results["redis"]="healthy"
            log_success "Redis is responding to ping"
        else
            health_results["redis"]="unhealthy"
            log_error "Redis is not responding to ping"
        fi
    else
        health_results["redis"]="unhealthy"
        log_error "Redis pod not found"
    fi
}

# Check resource usage
check_resource_usage() {
    log_info "Checking resource usage..."
    
    # Get node resource usage
    local nodes_info
    nodes_info=$(kubectl top nodes 2>/dev/null || echo "FAILED")
    
    if [ "$nodes_info" != "FAILED" ]; then
        log_info "Node resource usage:"
        echo "$nodes_info"
        
        # Check if any node is under high load
        local high_cpu_nodes
        high_cpu_nodes=$(echo "$nodes_info" | awk 'NR>1 && $3 ~ /[8-9][0-9]%|100%/ {print $1}' || echo "")
        
        if [ -n "$high_cpu_nodes" ]; then
            health_results["resources"]="degraded"
            log_warning "High CPU usage detected on nodes: $high_cpu_nodes"
        else
            health_results["resources"]="healthy"
            log_success "Node resource usage is normal"
        fi
    else
        health_results["resources"]="unknown"
        log_warning "Could not retrieve node resource usage (metrics-server may not be installed)"
    fi
    
    # Get pod resource usage
    local pods_info
    pods_info=$(kubectl top pods -n "$NAMESPACE" 2>/dev/null || echo "FAILED")
    
    if [ "$pods_info" != "FAILED" ]; then
        log_info "Pod resource usage in namespace $NAMESPACE:"
        echo "$pods_info"
    else
        log_warning "Could not retrieve pod resource usage"
    fi
}

# Check recent events
check_events() {
    log_info "Checking recent events..."
    
    local warning_events
    warning_events=$(kubectl get events -n "$NAMESPACE" --field-selector type=Warning --no-headers 2>/dev/null | wc -l || echo "0")
    
    if [ "$warning_events" -gt 0 ]; then
        health_results["events"]="degraded"
        log_warning "$warning_events warning events found in the last hour"
        
        log_info "Recent warning events:"
        kubectl get events -n "$NAMESPACE" --field-selector type=Warning --sort-by='.lastTimestamp' | tail -5
    else
        health_results["events"]="healthy"
        log_success "No recent warning events"
    fi
}

# Generate health report
generate_report() {
    echo
    log_info "=== HEALTH CHECK REPORT ==="
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $NAMESPACE"
    echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo
    
    local overall_status="healthy"
    local healthy_count=0
    local degraded_count=0
    local unhealthy_count=0
    local unknown_count=0
    
    # Print individual check results
    for check in "${!health_results[@]}"; do
        local status="${health_results[$check]}"
        local status_color=""
        
        case "$status" in
            "healthy")
                status_color="$GREEN"
                ((healthy_count++))
                ;;
            "degraded")
                status_color="$YELLOW"
                ((degraded_count++))
                if [ "$overall_status" = "healthy" ]; then
                    overall_status="degraded"
                fi
                ;;
            "unhealthy")
                status_color="$RED"
                ((unhealthy_count++))
                overall_status="unhealthy"
                ;;
            "unknown")
                status_color="$BLUE"
                ((unknown_count++))
                if [ "$overall_status" = "healthy" ]; then
                    overall_status="unknown"
                fi
                ;;
        esac
        
        printf "  %-20s ${status_color}%s${NC}\n" "$check:" "$status"
    done
    
    echo
    log_info "Summary:"
    echo "  Healthy: $healthy_count"
    echo "  Degraded: $degraded_count"
    echo "  Unhealthy: $unhealthy_count"
    echo "  Unknown: $unknown_count"
    
    echo
    case "$overall_status" in
        "healthy")
            log_success "Overall Status: HEALTHY"
            ;;
        "degraded")
            log_warning "Overall Status: DEGRADED"
            ;;
        "unhealthy")
            log_error "Overall Status: UNHEALTHY"
            ;;
        "unknown")
            log_info "Overall Status: UNKNOWN"
            ;;
    esac
    
    # Return appropriate exit code
    case "$overall_status" in
        "healthy") return 0 ;;
        "degraded") return 1 ;;
        "unhealthy") return 2 ;;
        "unknown") return 3 ;;
    esac
}

# Main function
main() {
    local check_type="${1:-all}"
    
    log_info "Starting health check for environment: $ENVIRONMENT"
    echo
    
    case "$check_type" in
        "cluster")
            check_cluster_connectivity
            ;;
        "pods")
            check_cluster_connectivity
            check_namespace
            check_pods
            ;;
        "deployments")
            check_cluster_connectivity
            check_namespace
            check_deployments
            ;;
        "services")
            check_cluster_connectivity
            check_namespace
            check_services
            ;;
        "api")
            check_cluster_connectivity
            check_namespace
            check_api_health
            ;;
        "database")
            check_cluster_connectivity
            check_namespace
            check_database
            ;;
        "redis")
            check_cluster_connectivity
            check_namespace
            check_redis
            ;;
        "resources")
            check_cluster_connectivity
            check_resource_usage
            ;;
        "all"|*)
            check_cluster_connectivity
            check_namespace
            check_pods
            check_deployments
            check_services
            check_ingress
            check_api_health
            check_database
            check_redis
            check_resource_usage
            check_events
            ;;
    esac
    
    echo
    generate_report
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            echo "AI Trading Platform Health Check Script"
            echo
            echo "Usage: $0 [check-type] [OPTIONS]"
            echo
            echo "Check Types:"
            echo "  all          Run all health checks (default)"
            echo "  cluster      Check Kubernetes cluster connectivity"
            echo "  pods         Check pod status"
            echo "  deployments  Check deployment status"
            echo "  services     Check service status"
            echo "  api          Check API health endpoint"
            echo "  database     Check database connectivity"
            echo "  redis        Check Redis connectivity"
            echo "  resources    Check resource usage"
            echo
            echo "Options:"
            echo "  -e, --environment ENV  Environment name (default: staging)"
            echo "  -n, --namespace NS     Kubernetes namespace (default: ai-trading-platform)"
            echo "  -t, --timeout SEC      Timeout for HTTP requests (default: 30)"
            echo "  -h, --help             Show this help message"
            echo
            echo "Exit Codes:"
            echo "  0  Healthy"
            echo "  1  Degraded"
            echo "  2  Unhealthy"
            echo "  3  Unknown"
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            exit 1
            ;;
        *)
            # First non-option argument is the check type
            if [ -z "${check_type:-}" ]; then
                check_type="$1"
            else
                log_error "Unexpected argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Run main function
main "${check_type:-all}"