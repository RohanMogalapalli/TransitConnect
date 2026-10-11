#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "=== Building TransitConnect ns-3 High-Speed Rail Simulator ==="

PKG_FLAGS=$(pkg-config --cflags --libs ns3-core ns3-network ns3-mobility ns3-propagation)
g++ -std=c++20 -O2 "$DIR/transitconnect_hsr_sim.cc" $PKG_FLAGS -o "$DIR/transitconnect_hsr_sim"

echo "=== Build Complete: $DIR/transitconnect_hsr_sim ==="
