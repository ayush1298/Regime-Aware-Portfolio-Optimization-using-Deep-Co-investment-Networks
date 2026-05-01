---
name: DeepCNL Web App
colors:
  primary: "#2c3e50"
  secondary: "#3498db"
  accent: "#e74c3c"
  success: "#27ae60"
  warning: "#f39c12"
  background: "#f8f9fa"
  surface: "#ffffff"
  on-surface: "#212529"
  on-surface-variant: "#6c757d"
  on-primary: "#ffffff"
  on-secondary: "#ffffff"
  surface-tint: "#e8f4f8"
typography:
  display:
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
    fontSize: "calc(1.475rem + 2.7vw)"
    fontWeight: "700"
  kpi-value:
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
    fontSize: "3rem"
    fontWeight: "700"
  kpi-label:
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
    fontSize: "1.2rem"
    fontWeight: "600"
  kpi-subtitle:
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
    fontSize: "0.9rem"
    fontWeight: "400"
rounded:
  sm: "8px"
  md: "10px"
spacing:
  sm: "0.5rem"
  md: "1rem"
  lg: "1.5rem"
  xl: "2rem"
  xxl: "4rem"
shadows:
  sm: "0 2px 4px rgba(0,0,0,0.1)"
  md: "0 4px 6px rgba(0,0,0,0.1)"
  hover: "0 2px 6px rgba(0,0,0,0.10)"
components:
  card:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.md}"
    padding: "{spacing.xl}"
    shadow: "{shadows.md}"
  panel:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.md}"
    padding: "{spacing.lg}"
    shadow: "{shadows.sm}"
  hero:
    background: "linear-gradient(135deg, {colors.primary}, {colors.secondary})"
    textColor: "{colors.on-primary}"
    padding: "{spacing.xxl} 0"
---

## Brand & Style
The DeepCNL web application adopts an **Analytical Professional** aesthetic. It aims to balance the academic rigor of deep learning and graph theory with the clarity required for financial dashboards. The design feels structured, data-forward, and clean, ensuring that complex network visualizations and backtest metrics remain the focal point.

## Colors
The color palette relies on high-contrast, semantic choices that guide the user's attention to critical performance indicators and actionable insights.

- **Primary (Dark Blue-Gray):** Used for typography headings and the foundational gradient of the hero section. It provides a grounded, authoritative feel.
- **Secondary (Light Blue):** Serves as the primary accent for active navigation, key data highlights, and interactive elements.
- **Semantic Colors:** Green, Orange, and Red are used strictly for conveying positive, neutral, or negative financial performance and regime states (e.g., Bull vs. Crisis).
- **Background:** A light off-white (`#f8f9fa`) ensures that white content cards stand out with subtle elevation.

## Typography
The system uses **Segoe UI** (with system fallbacks) to maintain a crisp, native feel across operating systems, which is expected in enterprise and financial tools.
- **KPI Emphasis:** Large, bold typography (3rem) is used for key metrics (Sharpe ratio, returns, hit ratios) to allow for quick scanning.
- **Hierarchy:** Clear distinction between KPI values, labels, and subdued subtitles (`#6c757d`) ensures dense information is easily digestible.

## Layout & Spacing
The layout leverages a responsive grid system based on Bootstrap 5.
- **Rhythm:** Generous padding (2rem inside cards, 4rem for hero sections) prevents the dense financial data and network graphs from feeling cluttered.
- **Containers:** Content is constrained within centered containers to maintain readability on ultra-wide monitors.

## Elevation & Depth
Depth is used sparingly to separate interactive containers from the background canvas.
- **Cards & Panels:** Subtle, low-opacity shadows (`rgba(0,0,0,0.1)`) create a soft lift. 
- **Hover States:** Interactive elements like KPI cards feature a subtle upward translation (`translateY(-5px)`) to provide tactile feedback without relying on heavy shadow expansion.

## Shapes
The shape language is slightly rounded but structural.
- **Corners:** A consistent `10px` border radius is applied to all main cards, graph containers, and panels. This softens the hard edges of data tables and canvas elements without making the UI feel informal. Smaller info boxes use an `8px` radius.
- **Borders:** Semantic border highlights (e.g., a 4px colored left border) are used to draw attention to specific KPI categories or predicted stock rows in tables, adding visual interest without cluttering the interface.
