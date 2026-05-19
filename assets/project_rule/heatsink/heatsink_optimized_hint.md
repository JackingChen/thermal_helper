PASSIVE THERMAL PRESET AVAILABLE: yes
Pre-computed preset = Step A: aluminum heatsink upgrade — CU-BASE, HP-1 to HP-5, and HP-xbar-1 to HP-xbar-5 change from copper to aluminum alloy.
TRIGGER: respond with {"action": "apply_optimized"} AND "mode_switch": "Thermal Simulation" when the user
confirms a material upgrade recommendation (e.g. says "Apply", "Switch to aluminum", "Yes", "Do it", "Optimize"
after an aluminum heatsink upgrade has been discussed or suggested in the conversation).
Do NOT apply on a general question — only on an explicit confirmation.
Do NOT generate move_sequence steps.
Explain in your response with this recommendation style:
**Recommendation:**
- Replace CU-BASE, HP-1 to HP-5, and HP-xbar-1 to HP-xbar-5 from copper C1100 to aluminum alloy.
- Aluminum reduces component weight by ~65 % with minimal conductivity trade-off at the system level.
- The PINN thermal model will switch to the aluminum-trained network, showing the updated fin temperature distribution (22–35°C range at the chip-inlet plane).
- This is recommended when weight reduction, cost, or vibration resistance is a design priority.
