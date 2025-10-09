#!/bin/bash

# List of scripts to run in order
scripts=(
    "run_spld_stage1.sh"
    "run_spld_stage1_2.sh"
    "run_spld_stage1_2_cs.sh"
    "run_spld_stage1_2_cs_stage3.sh"
)

# Stage names for display
stage_names=(
    "Stage 1 Training!"
    "Stage 2 Training!"
    "Cold Start!"
    "Stage 3 Training!"
)

# Run each script sequentially
for i in "${!scripts[@]}"; do
    echo "Running: ${stage_names[$i]}"
    
    # Make script executable if needed
    chmod +x "${scripts[$i]}"
    
    ./"${scripts[$i]}"
    
    # Check if script failed
    if [ $? -ne 0 ]; then
        echo "Error: ${scripts[$i]} failed"
        exit 1
    fi
    
    echo "Completed: ${stage_names[$i]}"
    echo "---"
done

echo "All stages completed successfully!"