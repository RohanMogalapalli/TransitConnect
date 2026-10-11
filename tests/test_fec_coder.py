"""
Unit tests for Systematic Block Erasure Coder and packet recovery.
"""

import pytest
from transitconnect.controller.dynamic_fec import SystematicBlockErasureCoder, Packet

def test_erasure_encoding_count():
    coder = SystematicBlockErasureCoder(block_size_k=10)
    data = [f"PAYLOAD_{i}".encode("utf-8") for i in range(10)]
    # Redundancy 0.30 -> ceil(10 * 0.3) = 3 parity packets
    encoded = coder.encode_block(data, redundancy_ratio=0.30)
    assert len(encoded) == 13
    assert sum(1 for p in encoded if not p.is_parity) == 10
    assert sum(1 for p in encoded if p.is_parity) == 3

def test_erasure_decoding_full_recovery():
    coder = SystematicBlockErasureCoder(block_size_k=10)
    data = [f"DATA_{i}".encode("utf-8") for i in range(10)]
    encoded = coder.encode_block(data, redundancy_ratio=0.30)  # 10 data + 3 parity = 13 packets

    # Simulate dropping 2 data packets (packets 2 and 4 are missing)
    # Total received = 11 packets (which is >= k=10)
    received = [p for i, p in enumerate(encoded) if i not in (2, 4)]
    assert len(received) == 11

    success, recovered, unrecoverable = coder.decode_block(received, k=10, total_transmitted=13)
    assert success is True
    assert recovered == 2
    assert unrecoverable == 0

def test_erasure_decoding_unrecoverable():
    coder = SystematicBlockErasureCoder(block_size_k=10)
    data = [f"DATA_{i}".encode("utf-8") for i in range(10)]
    encoded = coder.encode_block(data, redundancy_ratio=0.30)  # 10 data + 3 parity = 13 packets

    # Simulate severe loss: drop 5 data packets and 2 parity packets
    # Total received = 6 packets (< k=10)
    received = [p for i, p in enumerate(encoded) if i < 6]
    assert len(received) == 6

    success, recovered, unrecoverable = coder.decode_block(received, k=10, total_transmitted=13)
    assert success is False
    assert unrecoverable > 0
