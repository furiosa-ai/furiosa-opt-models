#!/usr/bin/env bash
set -euo pipefail

apt-get update -qq
apt-get install -yqq ibverbs-providers libcurl4 libibverbs1 libnuma1
