Demo script_04/20	
Scene 1：Best Placement Optimization	"- id: 1
  module: placement

  user_query: |
    請提供目前 layout 的最佳元件擺放建議，以優化散熱與氣流

  agent_response: |
    建議如下：
    1. 將 CPU 向左移動 5mm，以降低熱集中
    2. 將 FAN 向上移動 3mm，改善氣流入口
    3. 調整 VRM 與 CPU 間距至 8mm，以減少局部熱堆積

  user_query: |
    apply

  agent_placement:
    actions:
      - move_component: CPU, left, 5mm
      - move_component: FAN, up, 3mm
      - adjust_spacing: VRM-CPU, 8mm

  user_query: |
    3D

  agent_response: |
    已生成 3D 模型，請查看最佳化後的配置與氣流分佈"
Scene 2：Constraint-based Optimization	"- id: 2
  module: placement

  user_query: |
    在 FAN 固定於右側位置的情況下，CPU 可以如何調整以優化散熱？

  agent_response: |
    在 FAN 固定條件下，建議如下：
    1. 將 CPU 向左移動 6mm，以避免熱源集中於氣流出口
    2. 略微下移 CPU 2mm，使氣流可更均勻覆蓋散熱模組
    3. 調整 CPU 與散熱模組對齊，提高導熱效率

  user_query: |
    apply

  agent_placement:
    actions:
      - move_component: CPU, left, 6mm
      - move_component: CPU, down, 2mm
      - align: CPU with thermal_module

  user_query: |
    3D

  agent_response: |
    已生成 3D 模型，顯示在 FAN 固定條件下的最佳化配置"