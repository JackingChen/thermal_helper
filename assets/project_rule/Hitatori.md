# Hitatori — Fixed Layout Constraints

These constraints use `=` or an `_align` relation and **cannot be modified**.
All distances are in millimetres. "right/top/bottom" refers to the gap between
the named edge of `from` and the nearest edge of `to`.

## Board-edge anchors

| Component | Constraint | Meaning |
|---|---|---|
| hinge-L | left_align → edge_left | Left edge of hinge-L is flush with the left board edge |
| hinge-L | top_align → edge_top | Top edge of hinge-L is flush with the top board edge |
| hinge-cap | top_align → edge_top | Top edge of hinge-cap is flush with the top board edge |
| hinge-R | right_align → edge_right | Right edge of hinge-R is flush with the right board edge |
| hinge-R | top_align → edge_top | Top edge of hinge-R is flush with the top board edge |
| IO-L | left_align → edge_left | Left edge of IO-L is flush with the left board edge |
| IO-R | right_align → edge_right | Right edge of IO-R is flush with the right board edge |

## Hinge assembly (locked together)

| From | To | Constraint | Meaning |
|---|---|---|---|
| hinge-L | hinge-cap | right = 0 | hinge-L and hinge-cap are directly adjacent (0 mm gap) |
| hinge-cap | hinge-R | right = 0 | hinge-cap and hinge-R are directly adjacent (0 mm gap) |

## IO bracket alignment

| From | To | Constraint | Meaning |
|---|---|---|---|
| IO-L | hinge-L | top = 0 | IO-L top edge is flush with hinge-L top edge |
| IO-R | hinge-R | top = 0 | IO-R top edge is flush with hinge-R top edge |

## Fin / FAN assembly (locked together)

| From | To | Constraint | Meaning |
|---|---|---|---|
| Fin-1 | Fin-2 | top_align | Fin-1 and Fin-2 share the same top edge |
| Fin-1 | Fin-2 | right = 0 | Fin-1 and Fin-2 are directly adjacent (0 mm gap) |
| FAN-1 | Fin-1 | left_align | FAN-1 left edge is aligned with Fin-1 left edge |
| FAN-2 | Fin-2 | left_align | FAN-2 left edge is aligned with Fin-2 left edge |
| Fin-1 | FAN-1 | bottom = 1 | Fin-1 bottom edge is 1 mm above FAN-1 |
| Fin-2 | FAN-2 | bottom = 1 | Fin-2 bottom edge is 1 mm above FAN-2 |

## Vertical row offsets from hinge-cap

| From | To | Constraint | Meaning |
|---|---|---|---|
| hinge-cap | Fin-1 | bottom = 2 | Fin-1 top edge is 2 mm below the hinge-cap bottom edge |
| hinge-cap | CPU | bottom = 42 | CPU top edge is 42 mm below the hinge-cap bottom edge |

## CPU / DDR / Power-P alignment

| From | To | Constraint | Meaning |
|---|---|---|---|
| DDR-1 | CPU | top = 0 | DDR-1 top edge is flush with CPU top edge |
| CPU | DDR-1 | middle_x = 0 | CPU and DDR-1 are horizontally center-aligned |
| CPU | Power-P | top = 1 | Power-P top edge is 1 mm below CPU top edge |

---




