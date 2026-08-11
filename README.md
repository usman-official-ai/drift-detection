# Real-Time Data Drift Detection & Alert System

[![GitHub Actions](https://github.com/usman-official-ai/drift-detection/actions/workflows/drift_detection.yml/badge.svg)](https://github.com/usman-official-ai/drift-detection/actions)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready system that continuously monitors production data, detects distributional shifts (data drift) compared to training data, and sends real-time alerts via email/Slack.

## 🚀 Features

- **Multiple Detection Methods**
  - Kolmogorov-Smirnov (KS) test for numerical features
  - Population Stability Index (PSI) for categorical features
  - Customizable thresholds for each test

- **Data Pipeline**
  - Support for file, API, and database data sources
  - Automatic data validation and preprocessing
  - Historical data tracking

- **Alert System**
  - Email alerts via SMTP (Gmail support)
  - Slack notifications via webhook
  - Configurable alert thresholds

- **Automation**
  - Windows Task Scheduler integration
  - GitHub Actions CI/CD pipeline
  - Scheduled daily runs

- **Logging & Monitoring**
  - JSON-formatted structured logging
  - Rotating log files
  - Error tracking and reporting

- **Reporting**
  - HTML reports with visualizations
  - JSON reports for programmatic analysis
  - Historical drift tracking

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Drift Detection Methods](#drift-detection-methods)
- [Automation](#automation)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.10+
- Git
- Windows OS (for local scheduling) or GitHub account (for cloud automation)

### Step 1: Clone Repository

```bash
git clone https://github.com/usman-official-ai/drift-detection.git
cd drift-detection 
