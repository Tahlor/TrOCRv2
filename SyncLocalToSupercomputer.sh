#!/bin/bash

# Set the source and destination directories
SRC_DIR="/home/jclar234/TrOCR"
DEST_DIR="/home/jesse/TrOCR/ZZZ"

# Set the username and IP address of the remote computer
REMOTE_USER="jclar234"
REMOTE_IP="ssh.rc.byu.edu"

# Use rsync to sync the files between the two directories
rsync -avz --delete -e ssh ${REMOTE_USER}@${REMOTE_IP}:${SRC_DIR}/ ${DEST_DIR}/