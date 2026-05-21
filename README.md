# Dynamic Hotel Pricing Engine

This project implements an **experimental dynamic pricing engine** for hotels, designed to optimize room pricing based on demand, price elasticity, urgency, and inventory constraints. It combines parameter estimation from real hotel booking datasets, mathematical modeling, and Monte Carlo simulation to test pricing strategies and marketplace outcomes.

> **This is a Work In Progress – The model, outputs, and results do not yet reflect realistic hotel pricing or demand patterns. The numbers currently do not make sense and are for prototyping only.**

---

## Features

- **Pricing Parameter Estimator:** Estimates key pricing parameters (elasticity, urgency factor, demand, price bounds) from hotel booking data (uses [Kaggle Hotel Booking Demand dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)).
- **DynamicPricingEngine:** Core class modeling price optimization, demand/revenue functions, analytical gradients, and optimal price search.
- **MarketplaceSimulator:** Simulates a hotel marketplace over multiple days with customer agents, stochastic demand, and daily optimization.
- **Scenario Analysis & Visualization:** Includes main entry point, predefined scenarios, 30-day simulation, price/revenue trajectory visualizations, and strategy comparison.
- **Automated Tests:** (`test_engine.py`) for core model validation.

---

## Getting Started

1. **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repo-folder>
    ```

2. **Install dependencies (Python 3.8+):**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download hotel data (mandatory for parameter estimation):**
    - Get [hotel_bookings.csv](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) and place it in the repo folder.

4. **Estimate parameters (generates `hotel_pricing_parameters.json` and diagnostic plots):**
    ```bash
    python hotel_parameter_estimator.py
    ```

5. **Run main simulation/demo:**
    ```bash
    python main.py
    ```

6. **Run tests:**
    ```bash
    pytest test_engine.py
    ```

---

## Project Structure

- `engine.py` — Core pricing models, optimization, and simulation.
- `hotel_parameter_estimator.py` — Parameter and data-driven calibration.
- `hotel_config.py` — Loads hotel parameters from JSON.
- `main.py` — Interactive menu for scenario analysis and simulation runs.
- `test_engine.py` — Pytest-based test suite.

---

## Current Limitations & Issues

- **Number outputs are not yet realistic.** Output prices, revenue, and demand curves are not calibrated to produce sensible results.
- **Parameter estimation is basic.** More robust statistical fitting and validation needed.
- **Demand model is simple.** Doesn't handle multi-room bookings, group stays, or competition.
- **No real integration with property management systems (PMS), distribution systems, or competitor pricing feeds.**
- **Many behavioral/strategic edge cases unhandled.**
- **Customer simulation is simple and does not segment by business/leisure, booking channel, or purpose of travel.**

---

## Roadmap and Planned Improvements

**Phase 1: Model Calibration & Bug Fixes**
- [ ] Fix demand and revenue model math to ensure output numbers make sense.
- [ ] Calibrate parameter estimation (elasticity, urgency, demand) using real hotel booking data; add cross-validation, better error diagnostics, and unit conversions.
- [ ] Validate that optimal prices are within realistic bounds for a variety of hotel types.
- [ ] Add better checks and warnings when unrealistic parameter values or simulation results are detected.

**Phase 2: Model & Simulation Enhancements**
- [ ] Implement additional demand drivers (seasonality, weekday/weekend, events).
- [ ] Allow for multi-room bookings and group reservations.
- [ ] Support competitor rate effects and basic forecast blending.
- [ ] Add support for multi-segmented price strategies (business/leisure, OTA/direct).
- [ ] Simulate cancellations, overbooking, and re-optimization.
- [ ] Visualize booking curves vs. expected occupancy.

**Phase 3: Realism & Integration**
- [ ] Interface with real or simulated OTA/PMS data for real-time testing.
- [ ] Support for changing demand curves and parameters over time.
- [ ] Integrate external events (weather, city events).
- [ ] Simulate competitive response/price wars.

**Phase 4: UX and API**
- [ ] Build a simple web dashboard (Streamlit/Gradio/Flask).
- [ ] Implement REST API for integration with external tools.

**Stretch Goals**
- [ ] Support for personalized pricing by customer segment.
- [ ] Integrate with reinforcement learning or advanced RL optimization.
- [ ] Add A/B testing framework for pricing strategies.
- [ ] Simulation of multiple competing properties.

---


## License

[MIT License](LICENSE) – See file for details.
