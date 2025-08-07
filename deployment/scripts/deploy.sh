#!/bin/bash

# TPUv6-ZeroNAS Deployment Script
# Supports multiple deployment targets: local, docker, kubernetes

set -e

# Configuration
PROJECT_NAME="tpuv6-zeronas"
VERSION="0.1.0"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
NAMESPACE="${NAMESPACE:-default}"
DEPLOY_ENV="${DEPLOY_ENV:-development}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

show_usage() {
    echo "Usage: $0 [local|docker|kubernetes|cleanup] [options]"
    echo ""
    echo "Commands:"
    echo "  local      - Deploy locally using Python virtual environment"
    echo "  docker     - Deploy using Docker containers"
    echo "  kubernetes - Deploy to Kubernetes cluster"
    echo "  cleanup    - Clean up deployment artifacts"
    echo "  test       - Run deployment tests"
    echo ""
    echo "Options:"
    echo "  --env      - Deployment environment (development|production|minimal)"
    echo "  --registry - Docker registry URL"
    echo "  --namespace- Kubernetes namespace"
    echo "  --help     - Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  DEPLOY_ENV      - Deployment environment"
    echo "  DOCKER_REGISTRY - Docker registry URL"
    echo "  NAMESPACE       - Kubernetes namespace"
}

validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is required but not installed"
        exit 1
    fi
    
    log_success "Prerequisites validated"
}

deploy_local() {
    log_info "Starting local deployment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install package
    case $DEPLOY_ENV in
        "minimal")
            log_info "Installing minimal dependencies..."
            pip install -e .
            ;;
        "development")
            log_info "Installing development dependencies..."
            pip install -e ".[full,dev]"
            ;;
        "production")
            log_info "Installing production dependencies..."
            pip install -e ".[full]"
            ;;
        *)
            log_info "Installing with minimal dependencies (default)..."
            pip install -e .
            ;;
    esac
    
    # Run tests
    log_info "Running validation tests..."
    python scripts/simple_integration_test.py
    
    # Create start script
    cat > start_tpuv6_zeronas.sh << EOF
#!/bin/bash
cd \$(dirname \$0)
source venv/bin/activate
exec python -m tpuv6_zeronas.cli "\$@"
EOF
    chmod +x start_tpuv6_zeronas.sh
    
    log_success "Local deployment completed"
    log_info "Use './start_tpuv6_zeronas.sh --help' to get started"
}

deploy_docker() {
    log_info "Starting Docker deployment..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        exit 1
    fi
    
    # Build Docker image
    log_info "Building Docker image for $DEPLOY_ENV environment..."
    
    case $DEPLOY_ENV in
        "minimal")
            TARGET="minimal"
            ;;
        "development")
            TARGET="development"
            ;;
        "production")
            TARGET="production"
            ;;
        *)
            TARGET="production"
            ;;
    esac
    
    docker build \
        --target $TARGET \
        --tag "${PROJECT_NAME}:${VERSION}-${DEPLOY_ENV}" \
        --tag "${PROJECT_NAME}:latest-${DEPLOY_ENV}" \
        -f deployment/docker/Dockerfile .
    
    # Tag for registry if specified
    if [ "$DOCKER_REGISTRY" != "localhost:5000" ]; then
        docker tag "${PROJECT_NAME}:${VERSION}-${DEPLOY_ENV}" "${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}-${DEPLOY_ENV}"
        docker tag "${PROJECT_NAME}:latest-${DEPLOY_ENV}" "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest-${DEPLOY_ENV}"
        
        # Push to registry
        log_info "Pushing to registry: $DOCKER_REGISTRY"
        docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}-${DEPLOY_ENV}"
        docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest-${DEPLOY_ENV}"
    fi
    
    # Start services using docker-compose
    log_info "Starting Docker services..."
    cd deployment/docker
    
    # Export environment variables for docker-compose
    export TPUV6_VERSION="${VERSION}-${DEPLOY_ENV}"
    export TPUV6_ENV="$DEPLOY_ENV"
    
    case $DEPLOY_ENV in
        "development")
            docker-compose up -d tpuv6-zeronas-dev
            ;;
        "minimal")
            docker-compose up -d tpuv6-zeronas-minimal
            ;;
        *)
            docker-compose up -d tpuv6-zeronas-prod
            ;;
    esac
    
    cd ../..
    
    log_success "Docker deployment completed"
    log_info "Use 'docker-compose logs -f' to view logs"
}

deploy_kubernetes() {
    log_info "Starting Kubernetes deployment..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is required but not installed"
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy to Kubernetes
    log_info "Deploying to Kubernetes namespace: $NAMESPACE"
    
    # Update image in deployment
    sed -i.bak "s|image: tpuv6-zeronas:prod|image: ${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}-${DEPLOY_ENV}|g" \
        deployment/kubernetes/tpuv6-zeronas-deployment.yaml
    
    # Apply Kubernetes manifests
    kubectl apply -f deployment/kubernetes/ -n "$NAMESPACE"
    
    # Restore original deployment file
    mv deployment/kubernetes/tpuv6-zeronas-deployment.yaml.bak \
       deployment/kubernetes/tpuv6-zeronas-deployment.yaml
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/tpuv6-zeronas -n "$NAMESPACE" --timeout=300s
    
    # Get service information
    kubectl get services -n "$NAMESPACE"
    
    log_success "Kubernetes deployment completed"
    log_info "Use 'kubectl get pods -n $NAMESPACE' to check pod status"
    log_info "Use 'kubectl logs -f deployment/tpuv6-zeronas -n $NAMESPACE' to view logs"
}

cleanup_deployment() {
    log_info "Cleaning up deployment artifacts..."
    
    # Clean Docker
    if command -v docker &> /dev/null; then
        log_info "Cleaning Docker images..."
        docker rmi "${PROJECT_NAME}:${VERSION}-${DEPLOY_ENV}" 2>/dev/null || true
        docker rmi "${PROJECT_NAME}:latest-${DEPLOY_ENV}" 2>/dev/null || true
        
        # Stop docker-compose services
        if [ -f "deployment/docker/docker-compose.yml" ]; then
            cd deployment/docker
            docker-compose down --remove-orphans
            cd ../..
        fi
    fi
    
    # Clean Kubernetes
    if command -v kubectl &> /dev/null && kubectl cluster-info &> /dev/null; then
        log_info "Cleaning Kubernetes resources..."
        kubectl delete -f deployment/kubernetes/ -n "$NAMESPACE" --ignore-not-found=true
    fi
    
    # Clean local
    log_info "Cleaning local artifacts..."
    rm -rf venv/
    rm -f start_tpuv6_zeronas.sh
    rm -rf build/ dist/ *.egg-info/
    rm -f test_report.json
    
    log_success "Cleanup completed"
}

run_tests() {
    log_info "Running deployment tests..."
    
    # Test local installation
    log_info "Testing local installation..."
    python3 scripts/quick_test_minimal.py
    
    # Test Docker if available
    if command -v docker &> /dev/null; then
        log_info "Testing Docker image..."
        docker build --target minimal -t "${PROJECT_NAME}:test" -f deployment/docker/Dockerfile . --quiet
        docker run --rm "${PROJECT_NAME}:test"
        docker rmi "${PROJECT_NAME}:test" 2>/dev/null || true
    fi
    
    # Test Kubernetes manifests syntax if kubectl is available
    if command -v kubectl &> /dev/null; then
        log_info "Validating Kubernetes manifests..."
        kubectl apply --dry-run=client -f deployment/kubernetes/
    fi
    
    log_success "All deployment tests passed"
}

# Main script logic
case "$1" in
    "local")
        validate_prerequisites
        deploy_local
        ;;
    "docker")
        validate_prerequisites
        deploy_docker
        ;;
    "kubernetes")
        validate_prerequisites
        deploy_kubernetes
        ;;
    "cleanup")
        cleanup_deployment
        ;;
    "test")
        validate_prerequisites
        run_tests
        ;;
    "help"|"--help"|"-h")
        show_usage
        ;;
    "")
        log_error "No command specified"
        show_usage
        exit 1
        ;;
    *)
        log_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac

# Parse additional options
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            DEPLOY_ENV="$2"
            shift 2
            ;;
        --registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            log_warning "Unknown option: $1"
            shift
            ;;
    esac
done
