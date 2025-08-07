#!/bin/bash

# TPUv6-ZeroNAS Deployment Script
# Usage: ./deploy.sh [environment]

set -e

ENVIRONMENT=${1:-local}
VERSION=${VERSION:-latest}
IMAGE_NAME="tpuv6-zeronas"
REGISTRY=${REGISTRY:-""}

echo "🚀 Deploying TPUv6-ZeroNAS to ${ENVIRONMENT} environment"
echo "Version: ${VERSION}"

case $ENVIRONMENT in
  "local")
    echo "📦 Building Docker image locally..."
    docker build -t ${IMAGE_NAME}:${VERSION} .
    
    echo "🔧 Starting with Docker Compose..."
    docker-compose up -d
    
    echo "✅ Local deployment complete!"
    echo "Access the application at: http://localhost:8080"
    ;;
    
  "kubernetes"|"k8s")
    echo "🏗️ Building and pushing Docker image..."
    if [ -n "$REGISTRY" ]; then
      docker build -t ${REGISTRY}/${IMAGE_NAME}:${VERSION} .
      docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}
    else
      docker build -t ${IMAGE_NAME}:${VERSION} .
    fi
    
    echo "☸️ Deploying to Kubernetes..."
    kubectl apply -f deployment/kubernetes/
    
    echo "⏳ Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/tpuv6-zeronas
    
    echo "✅ Kubernetes deployment complete!"
    kubectl get pods -l app=tpuv6-zeronas
    ;;
    
  "production"|"prod")
    echo "🔐 Production deployment requires additional security checks"
    
    # Verify security scanning
    if command -v trivy &> /dev/null; then
      echo "🔍 Running security scan..."
      trivy image ${IMAGE_NAME}:${VERSION}
    fi
    
    # Build with production optimizations
    echo "🏗️ Building production image..."
    docker build \
      --build-arg PYTHON_ENV=production \
      --build-arg OPTIMIZE=true \
      -t ${REGISTRY}/${IMAGE_NAME}:${VERSION} \
      .
    
    if [ -n "$REGISTRY" ]; then
      docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}
    fi
    
    echo "☸️ Deploying to production Kubernetes..."
    kubectl apply -f deployment/kubernetes/
    kubectl set image deployment/tpuv6-zeronas tpuv6-zeronas=${REGISTRY}/${IMAGE_NAME}:${VERSION}
    
    echo "⏳ Rolling out update..."
    kubectl rollout status deployment/tpuv6-zeronas --timeout=600s
    
    echo "✅ Production deployment complete!"
    ;;
    
  *)
    echo "❌ Unknown environment: $ENVIRONMENT"
    echo "Available environments: local, kubernetes, production"
    exit 1
    ;;
esac

echo "🎉 Deployment to $ENVIRONMENT completed successfully!"