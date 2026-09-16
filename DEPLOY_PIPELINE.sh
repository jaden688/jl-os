#!/bin/bash
# Deployment pipeline: train -> export -> test -> deploy to kernel

set -e

MODEL_PT="${1:-mh_model.pt}"
BLOB_OUT="${2:-core.nhit}"

echo "==============================================================================="
echo " NHIT DEPLOYMENT PIPELINE"
echo "==============================================================================="

# Step 1: Export to binary blob
echo ""
echo "[1] Exporting ${MODEL_PT} to ${BLOB_OUT}..."
python export_to_blob.py "$MODEL_PT" --out "$BLOB_OUT"

# Step 2: Test on validation set
echo ""
echo "[2] Testing on validation set..."
gcc -O3 -std=c99 -DJL_HOST test_lm_host.c nhit.c jl_lm.c -o test_lm_host 2>/dev/null
./test_lm_host inference_standalone/lm_valset.bin "$BLOB_OUT"

# Step 3: Copy to standalone for quick testing
echo ""
echo "[3] Copying to inference_standalone..."
cp "$BLOB_OUT" inference_standalone/core.nhit

echo ""
echo "[4] Quick inference test..."
cd inference_standalone
gcc -O3 nhit.c jl_lm.c test.c -o test_inference 2>/dev/null
./test_inference core.nhit "the "
cd ..

# Step 4: Ready to deploy to kernel
echo ""
echo "==============================================================================="
echo " READY TO DEPLOY"
echo "==============================================================================="
echo ""
echo "The model ${BLOB_OUT} is ready."
echo ""
echo "Next steps:"
echo "  1. Update PEWTER-FS image with new model:"
echo "     cd inference_standalone && cp core.nhit /path/to/pewter.img"
echo ""
echo "  2. Rebuild kernel:"
echo "     make clean && make"
echo ""
echo "  3. Boot:"
echo "     make run-nographic"
echo ""
echo "  4. The kernel will load the new model and inference will run at startup"
echo ""
