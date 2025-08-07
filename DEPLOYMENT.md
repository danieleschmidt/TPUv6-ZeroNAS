# TPUv6-ZeroNAS Deployment Guide

This guide covers deployment options for TPUv6-ZeroNAS across different environments.

## Quick Start

### Minimal Installation (No Dependencies)

```bash
# Clone repository
git clone https://github.com/danieleschmidt/tpuv6-zeronas.git
cd tpuv6-zeronas

# Install with no external dependencies
pip install -e .

# Test installation
python scripts/quick_test_minimal.py

# Run basic search
python -m tpuv6_zeronas.cli search --max-iterations 10 --population-size 8
```

### Full Installation (With Scientific Libraries)

```bash
# Install with full dependencies
pip install -e ".[full]"

# Run comprehensive tests
python scripts/simple_integration_test.py
```

## Deployment Options

### 1. Local Development

```bash
# Deploy locally with virtual environment
./deployment/scripts/deploy.sh local --env development

# This creates a virtual environment and installs dependencies
# Use the generated script to run commands:
./start_tpuv6_zeronas.sh --help
```

### 2. Docker Deployment

```bash
# Build and run with Docker
./deployment/scripts/deploy.sh docker --env production

# Or use docker-compose directly:
cd deployment/docker
docker-compose up -d tpuv6-zeronas-prod
```

#### Available Docker Targets:
- `minimal`: Smallest possible image (~50MB)
- `production`: Production-ready with optimizations
- `development`: Full development environment with all tools

### 3. Kubernetes Deployment

```bash
# Deploy to Kubernetes cluster
./deployment/scripts/deploy.sh kubernetes --namespace tpuv6-nas

# Monitor deployment
kubectl get pods -n tpuv6-nas
kubectl logs -f deployment/tpuv6-zeronas -n tpuv6-nas
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|----------|
| `TPUV6_LOG_LEVEL` | Logging level | `INFO` |
| `TPUV6_OUTPUT_DIR` | Output directory | `./results` |
| `PYTHONPATH` | Python module path | `/app` |
| `DEPLOY_ENV` | Deployment environment | `development` |
| `DOCKER_REGISTRY` | Docker registry URL | `localhost:5000` |
| `NAMESPACE` | Kubernetes namespace | `default` |

## Configuration

### Search Configuration

Create `search-config.yaml`:

```yaml
max_iterations: 100
population_size: 20
target_tops_w: 75.0
max_latency_ms: 10.0
min_accuracy: 0.95
```

### Hardware Configuration

Create `hardware-config.yaml`:

```yaml
tpu_config:
  version: "v6"
  matrix_units: 4
  vector_units: 2
  peak_tops: 275.0
  memory_bandwidth_gbps: 900.0
```

## Performance Tuning

### Resource Requirements

| Deployment | CPU | Memory | Storage |
|------------|-----|--------|----------|
| Minimal | 0.1 core | 128MB | 100MB |
| Production | 1 core | 1GB | 1GB |
| Development | 2 cores | 2GB | 5GB |

### Scaling Guidelines

- **Small problems** (depth ≤ 8): 1 CPU core, 256MB RAM
- **Medium problems** (depth 8-15): 2 CPU cores, 1GB RAM  
- **Large problems** (depth > 15): 4+ CPU cores, 2GB+ RAM

## Monitoring

### Health Checks

```bash
# Local health check
python -c "from tpuv6_zeronas import ArchitectureSpace; print('healthy')"

# Docker health check
docker exec tpuv6-zeronas-prod python scripts/quick_test_minimal.py

# Kubernetes health check (automatic)
kubectl describe pods -l app=tpuv6-zeronas
```

### Logs

```bash
# Docker logs
docker logs tpuv6-zeronas-prod

# Kubernetes logs
kubectl logs -f deployment/tpuv6-zeronas -n tpuv6-nas

# Local logs
tail -f tpuv6_zeronas.log
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Solution: Use minimal installation
pip install -e .
# Verify with:
python scripts/quick_test_minimal.py
```

**Performance Issues**
```bash
# Check resource usage
top -p $(pgrep -f tpuv6-zeronas)

# Reduce search complexity
python -m tpuv6_zeronas.cli search --max-iterations 10 --population-size 5
```

**Docker Build Failures**
```bash
# Build minimal image only
docker build --target minimal -t tpuv6-zeronas:test .
```

### Debug Mode

```bash
# Enable debug logging
export TPUV6_LOG_LEVEL=DEBUG

# Run with detailed output
python -m tpuv6_zeronas.cli search --verbose
```

## Security Considerations

### Container Security
- Runs as non-root user
- Minimal attack surface (no external dependencies in minimal mode)
- Read-only filesystem where possible

### Network Security
- No network ports exposed by default
- All communication through standard input/output
- Optional Kubernetes network policies

### Data Security
- No sensitive data logged
- Results stored in configurable locations
- Support for encrypted storage volumes

## Backup and Recovery

### Data Backup
```bash
# Backup results
cp -r /path/to/results /backup/location/

# Backup configuration
cp *.yaml /backup/config/
```

### Disaster Recovery
```bash
# Clean installation
./deployment/scripts/deploy.sh cleanup
./deployment/scripts/deploy.sh local --env production

# Restore data
cp -r /backup/location/* ./results/
```

## Production Checklist

- [ ] Choose appropriate deployment target (minimal/production/development)
- [ ] Set resource limits and requests
- [ ] Configure logging and monitoring
- [ ] Set up health checks
- [ ] Plan backup strategy
- [ ] Test disaster recovery
- [ ] Security review completed
- [ ] Performance benchmarks established
- [ ] Documentation updated
- [ ] Team training completed

## Support

For deployment issues:
1. Check logs for error messages
2. Verify system requirements
3. Test with minimal configuration
4. Consult troubleshooting section
5. Create issue on GitHub with detailed information
