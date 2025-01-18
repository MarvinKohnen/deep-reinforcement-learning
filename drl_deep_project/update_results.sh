#!/bin/bash

cp -r scripts/our_agent/models scripts/our_agent/training_logs /tmp/
git checkout results
cp -r /tmp/models /tmp/training_logs scripts/our_agent/

# Commit and push changes on results branch
git add scripts/our_agent/models scripts/our_agent/training_logs
git commit -m "Update model and training logs"
git push origin results

# Switch back to marvin-private branch
git checkout marvin-private

echo "Model and training logs updated on results branch"