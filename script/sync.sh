#!/bin/bash
# Sync code to training server (excludes data files)
rsync -avz \
  --exclude='.git' --exclude='__pycache__' \
  --exclude-from="$(dirname "$0")/../.gitignore" \
  "$(dirname "$0")/../" \
  hdu-baiyang:/data2/wcn/UniCardio/
