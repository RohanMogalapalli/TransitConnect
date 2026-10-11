# TransitConnect

## Project Title

**A Handover-Risk-Aware Adaptive Sliding Window and Dynamic Forward Error Correction Protocol for High-Speed Mobile Networks**

## Overview

TransitConnect is a Computer Networks project focused on improving communication reliability in high-speed mobile environments such as bullet trains, metro systems, and connected vehicles. As a mobile node travels through cellular coverage regions, changing radio conditions and handovers can cause temporary increases in delay and packet loss.

TransitConnect combines network and mobility monitoring with machine-learning-based handover-risk prediction. The predicted risk is intended to help the communication protocol adapt its sliding-window behaviour and Forward Error Correction (FEC) redundancy to changing network conditions.

## Problem Statement

High-speed mobile devices frequently move between cellular coverage regions. During a handover, signal strength may deteriorate, round-trip time (RTT) may fluctuate, and packet loss may increase. Conventional sliding-window settings and statically configured FEC may react only after communication quality has degraded.

TransitConnect aims to anticipate potential handover-related degradation by estimating handover risk from current network conditions and mobility information. This risk estimate can then inform transmission-control decisions.

## Objectives

- Monitor network and mobility conditions in high-speed cellular environments.
- Estimate handover risk using multiple network and mobility parameters.
- Use the risk estimate to guide adaptive sliding-window behaviour.
- Adjust Forward Error Correction redundancy according to changing communication conditions.
- Improve communication reliability while managing delay, throughput, and coding overhead.
- Evaluate the protocol through controlled network simulations and comparisons with baseline approaches.

## System Architecture

The intended architecture consists of the following logical components:

1. **High-speed mobile node:** Represents a train, metro, or other mobile communication device.
2. **Cellular network:** Provides cellular coverage through base stations and the communication path to a remote host or server.
3. **Network and mobility monitoring:** Collects measurements that describe radio quality, delay, reliability, and mobility.
4. **Feature preparation:** Organises monitored measurements into a feature representation suitable for risk estimation.
5. **Handover-risk prediction:** Processes the features and produces a handover-risk score or risk state.
6. **Joint adaptive controller:** Uses the risk estimate to coordinate transmission adaptations.
7. **Adaptive sliding-window mechanism:** Adjusts sliding-window behaviour in response to the estimated risk.
8. **Dynamic FEC mechanism:** Adjusts error-correction redundancy according to communication conditions.
9. **Transmission feedback and evaluation:** Observes communication outcomes and supports analysis of protocol behaviour.

## Network and Mobility Parameters

TransitConnect uses six features to represent communication and mobility conditions:

- **Signal strength / signal condition:** Represents the current radio-link condition.
- **Signal fade rate:** Describes how quickly the signal condition is changing or deteriorating.
- **Round-trip time (RTT):** Represents communication delay.
- **RTT variance:** Captures variation in delay and communication instability.
- **Packet loss:** Represents unsuccessful packet delivery and reliability degradation.
- **Vehicle velocity:** Provides mobility context and indicates how quickly the node moves through cellular coverage regions.

Using these parameters together provides a broader view of network conditions than relying on a single measurement.

## Handover-Risk Prediction

The prediction component maps network and mobility observations to an estimate of the likelihood or risk of handover-related communication degradation. The input feature set consists of signal condition, signal fade rate, RTT, RTT variance, packet loss, and vehicle velocity.

The prediction output can be represented as a numerical risk score or as a conceptual category such as **Low Risk**, **Medium Risk**, or **High Risk**. The risk estimate provides information for the adaptive transmission controller.

A conceptual risk relationship is:

`H = w₁F + w₂V + w₃P + w₄R + w₅U`

Where:

- `H` is the handover-risk score.
- `F` is normalized signal fade rate.
- `V` is normalized RTT variance.
- `P` is normalized packet loss.
- `R` is normalized RTT condition.
- `U` is normalized vehicle velocity.
- `w₁` to `w₅` are weighting factors.

This equation expresses the conceptual relationship among the parameters; it should not be interpreted as a specific trained model unless the implementation defines it that way.

## Adaptive Sliding Window

The sliding-window mechanism governs how much data can be transmitted before acknowledgement or further feedback is required. TransitConnect intends to make this behaviour adaptive by using the predicted handover risk and observed network conditions. The aim is to respond to changing communication conditions rather than relying exclusively on fixed transmission settings.

## Dynamic Forward Error Correction

Forward Error Correction adds redundant information so that some transmission errors can be corrected without retransmitting every lost or corrupted packet. TransitConnect intends to vary FEC redundancy according to the predicted handover risk and communication conditions. The design aims to balance reliability improvements against the extra bandwidth and transmission overhead introduced by redundancy.

## Joint Adaptive Control

The joint controller coordinates sliding-window adaptation and dynamic FEC using the handover-risk estimate. Its purpose is to select complementary transmission adjustments based on the observed network state, rather than treating flow control and error correction as unrelated mechanisms.

## Simulation and Development Environment

The project uses **ns-3** as its network simulation platform and **Ubuntu Linux in a virtual machine** as the development environment. C++ is used for ns-3 simulation development, while Python supports machine-learning and data-processing tasks.

The simulation-based approach allows network behaviour, mobility, handover, and protocol parameters to be studied under controlled and repeatable conditions without requiring physical cellular infrastructure.

## Evaluation Metrics

The protocol can be evaluated using network and transmission metrics such as:

- Packet delivery ratio (PDR)
- Packet loss
- Throughput
- Average end-to-end delay
- Round-trip time (RTT)
- Handover interruption time
- FEC overhead
- Handover-risk score behaviour

Comparisons with baseline approaches can help assess the effect of the adaptive mechanisms on communication reliability and performance.

## Application Areas

Potential application areas include:

- High-speed trains and railway communication
- Metro systems
- Connected vehicles
- Intelligent Transportation Systems (ITS)
- Vehicle-to-Infrastructure (V2I) communication
- Other mobile environments where cellular handovers can affect communication reliability

