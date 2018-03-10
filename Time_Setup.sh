#!/usr/bin/env bash

echo 30 1 * * * "$(PWD)/DataProcess.py" | tee -a /var/spool/cron/root
