/*
 * TransitConnect: A Handover-Risk-Aware Adaptive Sliding Window and Dynamic FEC Protocol
 * ns-3 High-Speed Rail (HSR) Cellular Handover Simulation
 *
 * Implements a realistic multi-cell linear railway corridor with:
 * - 4-6 Macro cellular base stations (eNodeBs) with configurable spacing
 * - High-speed train UE mobility at configurable speeds (150 - 350 km/h)
 * - Distance-based path loss and log-normal shadow fading
 * - 3GPP A3 Handover event with configurable Hysteresis (dB) and Time-to-Trigger (TTT)
 * - Periodic telemetry export of the 6 core features:
 *   Signal strength (RSRP), Signal fade rate, RTT, RTT variance, Packet loss, Velocity
 */

#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/propagation-module.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <random>

using namespace ns3;

struct BaseStation {
    uint32_t id;
    Vector position;
    double txPowerDbm;
};

class TransitConnectHsrSim {
public:
    TransitConnectHsrSim();
    void Configure(int argc, char *argv[]);
    void Run();

private:
    void SetupTopology();
    void SampleTelemetry();
    double CalculateRsrp(const Vector& uePos, const Vector& bsPos, double txPowerDbm);

    // Simulation Parameters
    double m_speedKmh;
    uint32_t m_numBs;
    double m_bsSpacingM;
    double m_simTimeS;
    double m_sampleIntervalS;
    double m_hysteresisDb;
    double m_timeToTriggerS;
    double m_handoverExecDelayS;
    uint32_t m_seed;
    std::string m_traceOutputFile;

    // State Variables
    Ptr<Node> m_ueNode;
    std::vector<BaseStation> m_baseStations;
    uint32_t m_servingBsId;
    int32_t m_targetBsId;
    double m_tttTimer;
    bool m_isHandoverExecuting;
    double m_handoverExecTimer;
    uint32_t m_handoverCount;

    // Telemetry History
    double m_prevRsrp;
    double m_prevTime;
    double m_rttEwma;
    double m_rttVarEwma;
    std::vector<double> m_recentRtts;

    std::ofstream m_traceOut;
    std::mt19937 m_rng;
    std::normal_distribution<double> m_shadowingDist;
};

TransitConnectHsrSim::TransitConnectHsrSim()
    : m_speedKmh(300.0),
      m_numBs(5),
      m_bsSpacingM(2000.0),
      m_simTimeS(60.0),
      m_sampleIntervalS(0.05), // 50ms sampling
      m_hysteresisDb(3.0),
      m_timeToTriggerS(0.256), // 256ms 3GPP standard TTT
      m_handoverExecDelayS(0.080), // 80ms handover interruption
      m_seed(42),
      m_traceOutputFile("ns3_hsr_trace.csv"),
      m_servingBsId(0),
      m_targetBsId(-1),
      m_tttTimer(0.0),
      m_isHandoverExecuting(false),
      m_handoverExecTimer(0.0),
      m_handoverCount(0),
      m_prevRsrp(-70.0),
      m_prevTime(0.0),
      m_rttEwma(30.0),
      m_rttVarEwma(2.0),
      m_shadowingDist(0.0, 3.5) // Log-normal shadowing std dev 3.5 dB
{
}

void TransitConnectHsrSim::Configure(int argc, char *argv[]) {
    CommandLine cmd(__FILE__);
    cmd.AddValue("speed", "Train speed in km/h (e.g. 150, 250, 300, 350)", m_speedKmh);
    cmd.AddValue("numBs", "Number of cellular base stations", m_numBs);
    cmd.AddValue("bsSpacing", "Distance between base stations in meters", m_bsSpacingM);
    cmd.AddValue("simTime", "Simulation time in seconds", m_simTimeS);
    cmd.AddValue("sampleInterval", "Telemetry sampling interval in seconds", m_sampleIntervalS);
    cmd.AddValue("hysteresis", "Handover hysteresis margin in dB", m_hysteresisDb);
    cmd.AddValue("ttt", "Handover Time-to-Trigger in seconds", m_timeToTriggerS);
    cmd.AddValue("handoverDelay", "Handover execution interruption delay in seconds", m_handoverExecDelayS);
    cmd.AddValue("seed", "Random seed for reproducible fading", m_seed);
    cmd.AddValue("outputTrace", "Output CSV trace filename", m_traceOutputFile);
    cmd.Parse(argc, argv);

    m_rng.seed(m_seed);
    RngSeedManager::SetSeed(m_seed);
}

double TransitConnectHsrSim::CalculateRsrp(const Vector& uePos, const Vector& bsPos, double txPowerDbm) {
    double dx = uePos.x - bsPos.x;
    double dy = uePos.y - bsPos.y;
    double dz = uePos.z - bsPos.z;
    double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (distance < 1.0) distance = 1.0;

    // 3GPP HSR / Urban Macro Path Loss Model (approx. 2 GHz carrier)
    // PL(d) = 32.4 + 20*log10(f_GHz) + 30*log10(d_m)
    // Carrier freq = 2.1 GHz -> 32.4 + 20*log10(2.1) = 38.84 dB
    double pathLossDb = 38.84 + 30.0 * std::log10(distance);
    
    // Log-normal shadow fading component
    double shadowFadingDb = m_shadowingDist(m_rng);

    // Antenna gain (15 dBi eNodeB + 0 dBi UE)
    double antGainDb = 15.0;

    double rsrpDbm = txPowerDbm + antGainDb - pathLossDb + shadowFadingDb;
    return rsrpDbm;
}

void TransitConnectHsrSim::SetupTopology() {
    // 1. Create Base Stations along linear rail track (setback 50m, tower height 30m)
    for (uint32_t i = 0; i < m_numBs; ++i) {
        BaseStation bs;
        bs.id = i;
        bs.position = Vector(i * m_bsSpacingM, 50.0, 30.0);
        bs.txPowerDbm = 46.0; // 40 Watts
        m_baseStations.push_back(bs);
    }

    // 2. Create Train Node
    m_ueNode = CreateObject<Node>();
    Ptr<ConstantVelocityMobilityModel> mob = CreateObject<ConstantVelocityMobilityModel>();
    
    // Train starts 500m before the first base station
    mob->SetPosition(Vector(-500.0, 0.0, 3.0));
    
    // Convert speed km/h to m/s
    double speedMs = m_speedKmh * (1000.0 / 3600.0);
    mob->SetVelocity(Vector(speedMs, 0.0, 0.0));
    m_ueNode->AggregateObject(mob);

    // 3. Open Output CSV
    m_traceOut.open(m_traceOutputFile);
    if (!m_traceOut.is_open()) {
        std::cerr << "Error: Could not open output trace file: " << m_traceOutputFile << std::endl;
        exit(1);
    }

    // Header matching the 6-parameter monitoring requirements
    m_traceOut << "timestamp_s,train_x,velocity_kmh,serving_bs,target_bs,rsrp_dbm,"
               << "signal_fade_rate,rtt_ms,rtt_variance_ms,packet_loss_ratio,"
               << "is_handover,handover_count,interruption_time_ms\n";

    // Initial RSRP computation
    Vector initialPos = mob->GetPosition();
    m_prevRsrp = CalculateRsrp(initialPos, m_baseStations[0].position, m_baseStations[0].txPowerDbm);
    m_prevTime = 0.0;
}

void TransitConnectHsrSim::SampleTelemetry() {
    double now = Simulator::Now().GetSeconds();
    Ptr<ConstantVelocityMobilityModel> mob = m_ueNode->GetObject<ConstantVelocityMobilityModel>();
    Vector uePos = mob->GetPosition();

    // 1. Calculate RSRP from all base stations
    std::vector<double> rsrpList(m_numBs, -140.0);
    uint32_t bestBsId = 0;
    double bestRsrp = -140.0;

    for (uint32_t i = 0; i < m_numBs; ++i) {
        rsrpList[i] = CalculateRsrp(uePos, m_baseStations[i].position, m_baseStations[i].txPowerDbm);
        if (rsrpList[i] > bestRsrp) {
            bestRsrp = rsrpList[i];
            bestBsId = i;
        }
    }

    double servingRsrp = rsrpList[m_servingBsId];

    // 2. Signal fade rate: d(RSRP) / dt (dB/s)
    double dt = now - m_prevTime;
    double rawFadeRate = 0.0;
    if (dt > 1e-4) {
        // Negative delta means signal is fading
        rawFadeRate = (m_prevRsrp - servingRsrp) / dt;
    }
    m_prevRsrp = servingRsrp;
    m_prevTime = now;

    // 3. Handover A3 Event Evaluation (Best neighbor RSRP > Serving RSRP + Hysteresis)
    int32_t candidateTarget = -1;
    double maxTargetRsrp = -140.0;
    for (uint32_t i = 0; i < m_numBs; ++i) {
        if (i != m_servingBsId && rsrpList[i] > maxTargetRsrp) {
            maxTargetRsrp = rsrpList[i];
            candidateTarget = static_cast<int32_t>(i);
        }
    }

    bool isHandoverActive = false;
    double currentInterruptionMs = 0.0;

    if (m_isHandoverExecuting) {
        m_handoverExecTimer -= m_sampleIntervalS;
        isHandoverActive = true;
        currentInterruptionMs = m_sampleIntervalS * 1000.0;
        if (m_handoverExecTimer <= 0.0) {
            m_isHandoverExecuting = false;
            m_servingBsId = m_targetBsId;
            m_targetBsId = -1;
            m_handoverCount++;
        }
    } else {
        if (candidateTarget >= 0 && (maxTargetRsrp > servingRsrp + m_hysteresisDb)) {
            m_targetBsId = candidateTarget;
            m_tttTimer += m_sampleIntervalS;
            if (m_tttTimer >= m_timeToTriggerS) {
                // Trigger handover execution!
                m_isHandoverExecuting = true;
                m_handoverExecTimer = m_handoverExecDelayS;
                m_tttTimer = 0.0;
                isHandoverActive = true;
                currentInterruptionMs = m_handoverExecDelayS * 1000.0;
            }
        } else {
            // Reset TTT if condition drops below hysteresis
            m_tttTimer = 0.0;
            if (!m_isHandoverExecuting) {
                m_targetBsId = -1;
            }
        }
    }

    // 4. Packet Loss Ratio & RTT Dynamics
    // Base RTT ~ 25 ms. Increases with distance and spikes during cell edge / handover
    double baseRtt = 25.0;
    double distFromServing = std::abs(uePos.x - m_baseStations[m_servingBsId].position.x);
    double distFactor = distFromServing / m_bsSpacingM; // 0 to 1
    
    double lossRatio = 0.01; // baseline 1% channel error
    double instantaneousRtt = baseRtt + distFactor * 20.0;

    if (isHandoverActive) {
        // Severe degradation during handover execution
        lossRatio = 0.75 + (std::rand() % 25) / 100.0; // 75% - 100% loss during disruption
        instantaneousRtt += 180.0 + (std::rand() % 100); // 200+ ms RTT spike
    } else if (m_tttTimer > 0.0) {
        // Cell edge / impending handover degradation
        double tttProgress = m_tttTimer / m_timeToTriggerS;
        lossRatio = 0.05 + 0.35 * tttProgress; // Loss rises to ~40%
        instantaneousRtt += 40.0 * tttProgress;
    } else if (servingRsrp < -95.0) {
        lossRatio = 0.08;
        instantaneousRtt += 25.0;
    }

    // Add noise to RTT
    std::uniform_real_distribution<double> noiseDist(-3.0, 3.0);
    instantaneousRtt += noiseDist(m_rng);
    if (instantaneousRtt < 10.0) instantaneousRtt = 10.0;

    // EWMA for RTT and RTT Variance
    double alpha = 0.125;
    m_rttVarEwma = (1.0 - alpha) * m_rttVarEwma + alpha * std::abs(instantaneousRtt - m_rttEwma);
    m_rttEwma = (1.0 - alpha) * m_rttEwma + alpha * instantaneousRtt;

    // 5. Write to CSV
    m_traceOut << std::fixed << std::setprecision(4)
               << now << ","
               << uePos.x << ","
               << m_speedKmh << ","
               << m_servingBsId << ","
               << m_targetBsId << ","
               << std::setprecision(2)
               << servingRsrp << ","
               << rawFadeRate << ","
               << instantaneousRtt << ","
               << m_rttVarEwma << ","
               << std::setprecision(4)
               << lossRatio << ","
               << (isHandoverActive ? 1 : 0) << ","
               << m_handoverCount << ","
               << std::setprecision(2)
               << currentInterruptionMs << "\n";

    // Schedule next telemetry sample
    if (now + m_sampleIntervalS <= m_simTimeS) {
        Simulator::Schedule(Seconds(m_sampleIntervalS), &TransitConnectHsrSim::SampleTelemetry, this);
    }
}

void TransitConnectHsrSim::Run() {
    SetupTopology();

    // Schedule first sample
    Simulator::Schedule(Seconds(0.0), &TransitConnectHsrSim::SampleTelemetry, this);

    // Run ns-3 simulation loop
    Simulator::Stop(Seconds(m_simTimeS));
    Simulator::Run();
    Simulator::Destroy();

    m_traceOut.close();
    std::cout << "SUCCESS: ns-3 simulation completed. Trace saved to " << m_traceOutputFile
              << " (" << m_handoverCount << " handovers detected across " << m_numBs << " base stations)"
              << std::endl;
}

int main(int argc, char *argv[]) {
    TransitConnectHsrSim sim;
    sim.Configure(argc, argv);
    sim.Run();
    return 0;
}
