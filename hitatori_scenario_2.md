# Thermal Optimization Scenario 2: Hitatori Project (Intent Variation)

This document outlines an alternative user interaction flow for the **Hitatori project** thermal optimization scenario. It demonstrates that the Azure OpenAI backend relies on intent classification rather than rigid keyword matching, triggering the same system states using different phrasing.

## Sequence of Interaction

### 1. Initial Inquiry
**User:** *"Please give me thermal optimization suggestions for the CPU."*

* **Context State:** `preset_stage="initial"`
* **LLM Response:** The LLM interprets the request for optimization and, per its instructions for the initial stage, diagnoses thermal saturation. It recommends a **heatsink material swap** (changing `Fin-1` and `Fin-2` from Copper to Aluminum) for better transient heat response.
* **Action State Change:** `None` (or `null`). The LLM provides the recommendation and waits for explicit confirmation.

### 2. First Application
**User:** *"Do it."*

* **Context State:** `preset_stage="initial"`
* **LLM Response:** The LLM interprets "Do it." as explicit confirmation to apply the aluminum heatsink material swap.
* **Action State Change:** `("apply_optimized", "switch_thermal")`. 
  * The LLM outputs the JSON payload: `{"placement": {"action": "apply_optimized"}, "mode_switch": "Thermal Simulation"}`.
  * **Result:** The system triggers `_apply_optimized()`, loads `Hitatori_optimized.json`, switches the UI to the "Thermal Simulation" heatmap, and updates `preset_stage` to `"optimized"`.

### 3. Requesting Further Optimization
**User:** *"Refer to past projects with high-performance, thin-and-light thermal designs and propose a new suggestion."*

* **Context State:** `preset_stage="optimized"`
* **LLM Response:** This perfectly satisfies the LLM's prompt condition: *"When the user asks to improve further / refers to Project_X"*. Because the system is in the `"optimized"` stage, the backend forcefully injects a prompt requiring a specific response. The LLM replies with the mandated disclaimer:
  > *"Based on the current simulation result, I found that this case is outside the coverage of my training dataset. The model confidence is relatively low... I recommend collecting additional simulation data... Still, i can do somthing to the best of my knowledge. In my experience, Integrate 3× 6 mm sintered copper heat pipes (Heatsink-HP-1, 2, 3) bridging the CPU cold-plate directly to the rear fin stack."*
* **Action State Change:** `None` (or `null`). The AI recommends the heat pipes but waits for final confirmation.

### 4. Second Application
**User:** *"OK, let’s do it this way."*

* **Context State:** `preset_stage="optimized"`
* **LLM Response:** The LLM interprets "OK, let's do it this way." as an explicit confirmation to proceed with the heat pipe integration.
* **Action State Change:** `("apply_optimized_thermal", "switch_thermal")`. 
  * The LLM outputs the JSON payload: `{"placement": {"action": "apply_optimized_thermal"}, "mode_switch": "Thermal Simulation"}`.
  * **Result:** The system triggers `_apply_optimized_thermal()`, overwriting the board with `Hitatori_optimized_4_thermal.json`. The new heat pipes are simulated, the heatmap is updated, and the system advances to `preset_stage="optimized_thermal"`.
