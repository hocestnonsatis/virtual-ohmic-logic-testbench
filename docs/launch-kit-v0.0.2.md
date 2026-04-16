# VOLT v0.0.2 Launch Kit

Use these drafts as copy-paste starting points for launch posts.

## 1) Show HN draft

Title:
Show HN: VOLT - open-source analog crossbar inference simulator (C++17)

Body:
Hi HN - I built VOLT, a small open-source simulator for analog in-memory computing (AIMC) crossbars.

It lets you run voltage/conductance-based inference and measure how non-idealities affect output quality:
- ADC quantization limits
- thermal noise
- read disturb
- write endurance

The latest release (v0.0.2) adds rectangular N x M weight support, so it is easier to test real pretrained layer shapes.

Repo: https://github.com/hocestnonsatis/virtual-ohmic-logic-testbench
Release: https://github.com/hocestnonsatis/virtual-ohmic-logic-testbench/releases/tag/v0.0.2

I would love feedback on:
1) which non-idealities matter most in practical AIMC experiments
2) what result formats/plots would be most useful for research workflows

## 2) X (Twitter) draft

Built and released VOLT v0.0.2: an open-source C++17 simulator for analog crossbar inference.

It models how ADC quantization + thermal noise + read disturb + write endurance impact output error/SNR before hardware exists.

New: rectangular N x M weight support (up to 512 per dimension).

Repo:
https://github.com/hocestnonsatis/virtual-ohmic-logic-testbench

Release:
https://github.com/hocestnonsatis/virtual-ohmic-logic-testbench/releases/tag/v0.0.2

#AIMC #AnalogComputing #MLSystems #opensource

## 3) LinkedIn draft

I just published VOLT v0.0.2, an open-source simulator for Analog In-Memory Computing (AIMC) crossbar inference.

Why I built it:
- Evaluate analog inference behavior before hardware is available
- Quantify impact of practical non-idealities (ADC, noise, disturb, endurance)
- Keep experimentation fast and reproducible with a lightweight C++17 codebase

What is new in v0.0.2:
- Rectangular N x M matrix support for imported weights
- Updated multi-layer dimension validation
- Expanded tests and docs

Project:
https://github.com/hocestnonsatis/virtual-ohmic-logic-testbench

Release:
https://github.com/hocestnonsatis/virtual-ohmic-logic-testbench/releases/tag/v0.0.2

If you work on AIMC / hardware-aware ML, I would appreciate feedback and suggestions for the next milestones.
