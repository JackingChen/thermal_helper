# Thermal Optimization Scenario: Hitatori Project

This document outlines the user and AI assistant interaction flow for the **Hitatori project** thermal optimization scenario, detailing the expected Large Language Model (LLM) responses and the resulting system state changes when using the Azure OpenAI backend.

## Sequence of Interaction

### 1. Initial Inquiry
**User:** *"The CPU temperature exceeds the limit, help me optimize the thermal design."*

* **Context State:** `preset_stage="initial"`
* **LLM Response:** The LLM diagnoses thermal saturation and recommends a **heatsink material swap** (changing `Fin-1` and `Fin-2` from Copper C1100 to Aluminum 6061). It explains that aluminum's lower density reduces thermal mass, improving the heatsink's transient response to rapid temperature changes, potentially yielding a ~6 °C reduction in the CPU hotspot.
* **Action State Change:** `None` (or `null`). The LLM is strictly instructed **not** to apply presets on general questions. It provides the text recommendation and waits for explicit confirmation.

### 2. First Application
**User:** *"Go ahead and apply it."*

* **Context State:** `preset_stage="initial"`
* **LLM Response:** The LLM confirms that it is applying the recommended aluminum heatsink material swap.
* **Action State Change:** `("apply_optimized", "switch_thermal")`. 
  * Because the user gave an explicit confirmation ("apply"), the LLM outputs the JSON payload: `{"placement": {"action": "apply_optimized"}, "mode_switch": "Thermal Simulation"}`.
  * **Result:** The system triggers `_apply_optimized()`, which overwrites the board positions with `Hitatori_optimized.json`, clears the simulation data, and immediately switches the workspace UI to the "Thermal Simulation" heatmap view. The internal `preset_stage` updates to `"optimized"`.

### 3. Requesting Further Optimization
**User:** *"Based on existing heat pipe / dual-zone heat spreading design cases, give me a better solution than the current one."*

* **Context State:** `preset_stage="optimized"`
* **LLM Response:** Because the current `preset_stage` is `"optimized"`, the backend injects a specific prompt. The LLM will reply closely or exactly with:
  > *"Based on the current simulation result, I found that this case is outside the coverage of my training dataset. The model confidence is relatively low, so the prediction may not be reliable enough for engineering decision-making. I recommend collecting additional simulation data under similar thermal conditions and retraining the model to improve accuracy and coverage.*
  > *Still, i can do somthing to the best of my knowledge. In my experience, Integrate 3× 6 mm sintered copper heat pipes (Heatsink-HP-1, 2, 3) bridging the CPU cold-plate directly to the rear fin stack."*
* **Action State Change:** `None` (or `null`). It again waits for explicit user confirmation before modifying the board.

### 4. Second Application
**User:** *"Apply."*

* **Context State:** `preset_stage="optimized"`
* **LLM Response:** The LLM confirms that it is integrating the 3× 6 mm sintered copper heat pipes to the layout to bridge the cold-plate and fin stack.
* **Action State Change:** `("apply_optimized_thermal", "switch_thermal")`. 
  * Triggered by the explicit confirmation, the LLM outputs the JSON payload: `{"placement": {"action": "apply_optimized_thermal"}, "mode_switch": "Thermal Simulation"}`.
  * **Result:** The system triggers `_apply_optimized_thermal()`, which overwrites the board positions with `Hitatori_optimized_4_thermal.json`. This adds the new copper heat pipe components to the simulation, triggers a re-render of the thermal heatmap, and advances the internal `preset_stage` to `"optimized_thermal"`.
