# Insider Threat Detection System

## Overview

The Insider Threat Detection System is a web-based cybersecurity platform designed to identify anomalous user behavior within enterprise environments. It leverages User and Entity Behavioral Analytics (UEBA) techniques combined with machine learning models to detect potential insider threats in near real-time.

This system is built as part of a capstone project and focuses on providing interpretable, actionable insights for security analysts through a streamlined dashboard interface.

---

## Key Capabilities

* Real-time detection of anomalous user behavior
* AI-driven analysis using Autoencoder and Isolation Forest models
* Structured alert generation with risk prioritization
* Interactive dashboard for monitoring and investigation
* Explainable insights with drill-down analysis
* Simulated live data streaming for realistic testing

---

## System Architecture

The platform is composed of the following core modules:

1. **Data Ingestion Module**

   * Loads historical data and simulates real-time log streaming

2. **Feature Engineering Module**

   * Transforms raw logs into normalized behavioral feature vectors

3. **Autoencoder Module**

   * Learns compressed representations of user behavior

4. **Isolation Forest Module**

   * Detects anomalies based on learned behavior patterns

5. **Alert Generation Engine**

   * Produces prioritized alerts based on anomaly scores

6. **Explainability Module**

   * Provides detailed insights into why anomalies are flagged

7. **Streamlit Dashboard**

   * Visual interface for monitoring alerts and conducting investigations

---

## How It Works

1. The system ingests structured log data from the CERT dataset
2. Behavioral features are extracted and normalized
3. The Autoencoder compresses data into latent embeddings
4. The Isolation Forest evaluates embeddings for anomalies
5. Alerts are generated when suspicious behavior is detected
6. Analysts can investigate alerts through the dashboard with detailed explanations

---

## Technology Stack

* **Programming Language:** Python 3.x
* **Frontend:** Streamlit
* **Machine Learning:** TensorFlow/Keras (Autoencoder), Scikit-learn (Isolation Forest)
* **Data Processing:** NumPy, Pandas

---

## Operating Environment

* Runs as a web-based application (local or hosted)
* Minimum 8GB RAM recommended
* Works on standard development machines
* Simulated streaming environment (no live enterprise integration)

---

## Key Features

* Real-time anomaly scoring
* Risk-based alert prioritization
* Interactive visualizations
* Drill-down analysis to feature-level contributions
* Continuous streaming simulation

---

## Limitations

* Uses simulated data (CERT dataset)
* No integration with production SIEM systems
* No automated incident response
* Designed for academic and demonstration purposes

---

## Security & Compliance Considerations

* Processes data locally (no external transmission)
* Alerts are probabilistic and require analyst review
* Designed with audit logging and traceability in mind
* Would require compliance adjustments for real-world deployment

---

## Future Enhancements

* Integration with enterprise SIEM platforms
* Real-time production data ingestion
* Advanced model tuning and additional ML techniques
* Automated response mechanisms

---

## Team

* Aaron Lorea - https://www.linkedin.com/in/aaronloera324/
* Tyler Kees - https://www.linkedin.com/in/tyler-kees/
* Melusi Senzanje - https://www.linkedin.com/in/melusisenzanje/
* Melody Nnadi - https://www.linkedin.com/in/melodynnadi/
* Hugo Margues - https://www.linkedin.com/in/hugomarquesnob/
* Matthew Emanuel - https://www.linkedin.com/in/matthew-emanuel-1b168a340/

---

## License

This project is developed for academic purposes. Use of third-party libraries must comply with their respective licenses.

---

## Acknowledgments

* CERT Insider Threat Dataset (Carnegie Mellon University)
* TensorFlow, Scikit-learn, and Streamlit communities
