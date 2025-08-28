#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# One-click script to run MLA disaggregated prefill/decode example
# This script launches all three components (decoder, prefiller, proxy) and manages their lifecycle

set -e  # Exit on error

# Configuration
MODEL="deepseek-ai/DeepSeek-V2-Lite"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${SCRIPT_DIR}/.mla_pids"

# Service ports
PREFILLER_PORT=8100
DECODER_PORT=8200
PROXY_PORT=8000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for a service to be ready
wait_for_service() {
    local service_name=$1
    local port=$2
    local max_wait=120  # Maximum wait time in seconds
    local elapsed=0
    
    print_info "Waiting for $service_name to be ready on port $port..."
    
    while [ $elapsed -lt $max_wait ]; do
        if check_port $port; then
            # Additional check for actual HTTP response
            if curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health 2>/dev/null | grep -q "200\|404"; then
                print_info "$service_name is ready!"
                return 0
            fi
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        echo -n "."
    done
    
    print_error "$service_name failed to start within $max_wait seconds"
    return 1
}

# Function to cleanup processes
cleanup() {
    print_info "Cleaning up..."
    
    if [ -f "$PID_FILE" ]; then
        while IFS= read -r pid; do
            if kill -0 $pid 2>/dev/null; then
                print_info "Stopping process $pid"
                kill -TERM $pid 2>/dev/null || true
                sleep 1
                kill -KILL $pid 2>/dev/null || true
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
    
    # Also kill any remaining vllm processes
    pkill -f "vllm serve.*$MODEL" 2>/dev/null || true
    pkill -f "proxy_server.py" 2>/dev/null || true
    
    print_info "Cleanup complete"
}

# Function to start a service
start_service() {
    local service_name=$1
    local command=$2
    local log_file=$3
    
    print_info "Starting $service_name..."
    
    # Start the service in background and capture PID
    nohup bash -c "$command" > "$log_file" 2>&1 &
    local pid=$!
    
    # Save PID for cleanup
    echo $pid >> "$PID_FILE"
    
    print_info "$service_name started with PID $pid"
    print_info "Logs: $log_file"
    
    # Give it a moment to start
    sleep 2
    
    # Check if process is still running
    if ! kill -0 $pid 2>/dev/null; then
        print_error "$service_name failed to start. Check logs at $log_file"
        tail -20 "$log_file"
        return 1
    fi
    
    return 0
}

# Main execution
main() {
    print_info "=========================================="
    print_info "MLA Disaggregated Inference Example"
    print_info "Model: $MODEL"
    print_info "=========================================="
    
    # Check prerequisites
    print_info "Checking prerequisites..."
    
    # Check if vLLM is installed
    if ! command -v vllm &> /dev/null; then
        print_error "vLLM is not installed. Please install it first."
        exit 1
    fi
    
    # Check if LMCache is properly installed
    if ! python -c "import lmcache" 2>/dev/null; then
        print_error "LMCache is not installed. Please install it first."
        exit 1
    fi
    
    # Check for required ports
    for port in $PREFILLER_PORT $DECODER_PORT $PROXY_PORT; do
        if check_port $port; then
            print_error "Port $port is already in use. Please free it or change the configuration."
            exit 1
        fi
    done
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Clean up any previous runs
    cleanup
    
    # Trap to ensure cleanup on exit
    trap cleanup EXIT INT TERM
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Set environment variables
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    # For high concurrency, optionally disable cloning
    if [ "${ENABLE_HIGH_CONCURRENCY:-0}" = "1" ]; then
        export LMCACHE_NIXL_ENABLE_CLONE=0
        print_warn "High concurrency mode enabled - tensor cloning disabled"
    fi
    
    # Start Decoder Service
    DECODER_CMD="LMCACHE_CONFIG_FILE=configs/lmcache-decoder-config.yaml \
        CUDA_VISIBLE_DEVICES=2,3 \
        vllm serve $MODEL \
        --host localhost \
        --port $DECODER_PORT \
        --tensor-parallel-size 2 \
        --max-model-len 32768 \
        --max-num-batched-tokens 32768 \
        --max-num-seqs 256 \
        --enforce-eager \
        --disable-log-requests \
        --trust-remote-code \
        --dtype auto"

        # --kv-connector lmcache \
        # --kv-role reuse \
        # --kv-transfer-mode decode \

    if ! start_service "Decoder" "$DECODER_CMD" "$LOG_DIR/decoder.log"; then
        print_error "Failed to start Decoder service"
        exit 1
    fi
    
    # Wait for decoder to be ready
    if ! wait_for_service "Decoder" $DECODER_PORT; then
        print_error "Decoder service did not become ready"
        exit 1
    fi
    
    # Start Prefiller Service
    PREFILLER_CMD="LMCACHE_CONFIG_FILE=configs/lmcache-prefiller-config.yaml \
        CUDA_VISIBLE_DEVICES=0,1 \
        vllm serve $MODEL \
        --host localhost \
        --port $PREFILLER_PORT \
        --tensor-parallel-size 2 \
        --max-model-len 32768 \
        --max-num-batched-tokens 32768 \
        --max-num-seqs 256 \
        --enforce-eager \
        --disable-log-requests \
        --trust-remote-code \
        --dtype auto"
    
        # --kv-connector lmcache \
        # --kv-role fill \
        # --kv-transfer-mode prefill \

    if ! start_service "Prefiller" "$PREFILLER_CMD" "$LOG_DIR/prefiller.log"; then
        print_error "Failed to start Prefiller service"
        exit 1
    fi
    
    # Wait for prefiller to be ready
    if ! wait_for_service "Prefiller" $PREFILLER_PORT; then
        print_error "Prefiller service did not become ready"
        exit 1
    fi
    
    # Start Proxy Server
    PROXY_CMD="python proxy_server.py \
        --host localhost \
        --port $PROXY_PORT \
        --prefiller-host localhost \
        --prefiller-port $PREFILLER_PORT \
        --decoder-host localhost \
        --decoder-port $DECODER_PORT \
        --nixl-receiver-host localhost \
        --nixl-receiver-port 55555"
    
    if ! start_service "Proxy" "$PROXY_CMD" "$LOG_DIR/proxy.log"; then
        print_error "Failed to start Proxy server"
        exit 1
    fi
    
    # Wait for proxy to be ready
    if ! wait_for_service "Proxy" $PROXY_PORT; then
        print_error "Proxy server did not become ready"
        exit 1
    fi
    
    print_info "=========================================="
    print_info "All services started successfully!"
    print_info "=========================================="
    print_info "Proxy URL: http://localhost:$PROXY_PORT"
    print_info "Logs directory: $LOG_DIR"
    print_info ""
    print_info "To test the setup, run:"
    print_info "  python test_mla_inference.py"
    print_info ""
    print_info "To run concurrent tests:"
    print_info "  python test_mla_inference.py --test-concurrent"
    print_info ""
    print_info "To monitor logs:"
    print_info "  tail -f $LOG_DIR/proxy.log"
    print_info ""
    print_info "Press Ctrl+C to stop all services"
    print_info "=========================================="
    
    # Optional: Run a test request
    if [ "${RUN_TEST:-0}" = "1" ]; then
        sleep 5
        print_info "Running test request..."
        python test_mla_inference.py
    fi
    
    # Wait for user interrupt
    print_info "Services are running. Press Ctrl+C to stop..."
    
    # Keep the script running and monitor services
    while true; do
        sleep 5
        
        # Check if all services are still running
        all_running=true
        while IFS= read -r pid; do
            if ! kill -0 $pid 2>/dev/null; then
                print_error "Process $pid has stopped unexpectedly"
                all_running=false
                break
            fi
        done < "$PID_FILE"
        
        if [ "$all_running" = false ]; then
            print_error "One or more services have stopped. Shutting down..."
            break
        fi
    done
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --high-concurrency)
            export ENABLE_HIGH_CONCURRENCY=1
            shift
            ;;
        --run-test)
            export RUN_TEST=1
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --high-concurrency  Enable high concurrency mode (disables tensor cloning)"
            echo "  --run-test         Run a test request after starting services"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main