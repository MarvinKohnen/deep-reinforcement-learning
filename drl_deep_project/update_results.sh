#!/bin/bash

# Prompt the user for a message
echo "Please enter a short message describing the training run:"
read user_message

# Copy files to temporary location
cp -r scripts/our_agent/models scripts/our_agent/training_logs /tmp/

# Switch to results branch
git checkout results

# Copy files back
cp -r /tmp/models /tmp/training_logs scripts/our_agent/

# Find the most recently modified directory in training_logs
latest_dir=$(find scripts/our_agent/training_logs -maxdepth 1 -type d -printf '%T@ %P\n' | sort -n | tail -1 | cut -f2- -d" ")

# Create README with user message in the latest directory
echo "$user_message" > "scripts/our_agent/training_logs/$latest_dir/README.md"

# Commit and push changes on results branch
git add scripts/our_agent/models scripts/our_agent/training_logs
git commit -m "Update model and training logs"
git push origin results

<<<<<<< HEAD
# Switch back to marvin-private branch
git checkout marvin-private
=======
# Switch back to julius-private branch
git checkout julius-private
>>>>>>> julius-private

echo "Model and training logs updated on results branch with README in $latest_dir"