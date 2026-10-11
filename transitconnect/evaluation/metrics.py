"""
Networking Evaluation Metrics Engine for TransitConnect.
Calculates:
- Packet Delivery Ratio (PDR, %)
- Effective Goodput (Mbps) and Raw Throughput (Mbps)
- Average End-to-End Delay (ms) and RTT Variance
- Handover Interruption Time (ms)
- FEC Redundancy Overhead (%)
"""

from typing import List, Dict, Any
import numpy as np

def calculate_pdr(delivered_data: int, sent_data: int) -> float:
    """Calculates Packet Delivery Ratio (PDR) as a percentage."""
    if sent_data <= 0:
        return 100.0
    return round((delivered_data / sent_data) * 100.0, 2)

def calculate_goodput_mbps(delivered_data: int, total_duration_s: float, packet_size_bytes: int = 1400) -> float:
    """Calculates effective Goodput in Megabits per second (useful delivered data only)."""
    if total_duration_s <= 0:
        return 0.0
    total_bits = delivered_data * packet_size_bytes * 8.0
    return round(total_bits / (total_duration_s * 1e6), 3)

def calculate_throughput_mbps(total_sent_packets: int, total_duration_s: float, packet_size_bytes: int = 1400) -> float:
    """Calculates total raw throughput including parity bytes in Mbps."""
    if total_duration_s <= 0:
        return 0.0
    total_bits = total_sent_packets * packet_size_bytes * 8.0
    return round(total_bits / (total_duration_s * 1e6), 3)

def calculate_fec_overhead_percent(sent_parity: int, total_sent: int) -> float:
    """Calculates FEC overhead as a percentage of total transmitted bandwidth."""
    if total_sent <= 0:
        return 0.0
    return round((sent_parity / total_sent) * 100.0, 2)

def aggregate_protocol_metrics(
    step_records: List[Dict[str, Any]],
    sim_time_s: float,
    speed_kmh: float,
    protocol_name: str
) -> Dict[str, Any]:
    """Aggregates per-monitoring-step records into a standardized evaluation summary."""
    total_sent_data = sum(r.get("sent_data", 0) for r in step_records)
    total_sent_parity = sum(r.get("sent_parity", 0) for r in step_records)
    total_sent = total_sent_data + total_sent_parity
    total_delivered = sum(r.get("delivered_data", 0) for r in step_records)
    total_lost = sum(r.get("lost_unrecovered", 0) for r in step_records)
    total_recovered = sum(r.get("recovered_by_fec", 0) for r in step_records)

    rtts = [r.get("rtt_ms", 30.0) for r in step_records]
    delays = [r.get("rtt_ms", 30.0) / 2.0 for r in step_records]  # One-way delay approx half RTT
    interruptions = [r.get("interruption_time_ms", 0.0) for r in step_records]

    pdr = calculate_pdr(total_delivered, total_sent_data)
    goodput = calculate_goodput_mbps(total_delivered, sim_time_s)
    throughput = calculate_throughput_mbps(total_sent, sim_time_s)
    fec_overhead = calculate_fec_overhead_percent(total_sent_parity, total_sent)

    return {
        "protocol": protocol_name,
        "speed_kmh": speed_kmh,
        "sim_time_s": round(sim_time_s, 2),
        "total_sent_data_packets": total_sent_data,
        "total_sent_parity_packets": total_sent_parity,
        "total_sent_packets": total_sent,
        "delivered_data_packets": total_delivered,
        "lost_unrecovered_packets": total_lost,
        "recovered_by_fec_packets": total_recovered,
        "pdr_percent": pdr,
        "packet_loss_percent": round(100.0 - pdr, 2),
        "goodput_mbps": goodput,
        "raw_throughput_mbps": throughput,
        "avg_e2e_delay_ms": round(float(np.mean(delays)), 2) if delays else 0.0,
        "avg_rtt_ms": round(float(np.mean(rtts)), 2) if rtts else 0.0,
        "p95_rtt_ms": round(float(np.percentile(rtts, 95)), 2) if rtts else 0.0,
        "total_interruption_ms": round(float(np.sum(interruptions)), 2),
        "fec_overhead_percent": fec_overhead
    }
