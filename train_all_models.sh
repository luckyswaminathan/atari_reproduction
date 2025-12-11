#!/bin/bash
# Script to train all DQN model variants sequentially

set -e  # Exit on error (comment out if you want to continue after failures)

MODELS=("base" "dueling" "mha" "dueling_mha")
TOTAL=${#MODELS[@]}

echo "============================================================"
echo "DQN Model Training Suite"
echo "============================================================"
echo "Models to train: ${MODELS[*]}"
echo "Total models: $TOTAL"
echo "Start time: $(date)"
echo "============================================================"
echo ""

OVERALL_START=$(date +%s)

for i in "${!MODELS[@]}"; do
    MODEL=${MODELS[$i]}
    NUM=$((i + 1))
    
    echo ""
    echo "[$NUM/$TOTAL] Starting training for model: $MODEL"
    echo "============================================================"
    
    START=$(date +%s)
    
    # Run training
    if uv run python train.py --model "$MODEL"; then
        ELAPSED=$(($(date +%s) - START))
        HOURS=$((ELAPSED / 3600))
        MINS=$(((ELAPSED % 3600) / 60))
        SECS=$((ELAPSED % 60))
        
        echo ""
        echo "✓ Completed training for model: $MODEL"
        echo "  Duration: ${HOURS}h ${MINS}m ${SECS}s"
    else
        ELAPSED=$(($(date +%s) - START))
        echo ""
        echo "✗ Training failed for model: $MODEL"
        echo "  Duration before failure: ${ELAPSED}s"
        # Uncomment next line if you want to stop on failure:
        # exit 1
    fi
    
    echo "============================================================"
    
    # Wait before next model (except for last one)
    if [ $NUM -lt $TOTAL ]; then
        echo ""
        echo "Waiting 5 seconds before starting next model..."
        sleep 5
    fi
done

OVERALL_ELAPSED=$(($(date +%s) - OVERALL_START))
HOURS=$((OVERALL_ELAPSED / 3600))
MINS=$(((OVERALL_ELAPSED % 3600) / 60))
SECS=$((OVERALL_ELAPSED % 60))

echo ""
echo "============================================================"
echo "Training Suite Complete"
echo "============================================================"
echo "Total duration: ${HOURS}h ${MINS}m ${SECS}s"
echo "End time: $(date)"
echo "============================================================"

