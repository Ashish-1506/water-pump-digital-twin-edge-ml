"""
generate_architecture_diagram.py
================================
Generates a publication-grade research architecture diagram for the
Cyber-Physical TinyML Water Pump Fault Diagnostics System.
Outputs high-resolution 300 DPI PNG and vector SVG files.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = Path("ML/model_output/report_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling settings
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

fig = plt.figure(figsize=(18, 11), facecolor="#F8FAFC")
ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Theme Palette (Modern Academic / Industrial Cyber-Physical)
C_TITLE_BG = "#0F172A"       # Slate 900
C_LAYER_BG = "#FFFFFF"       # Pure White
C_BORDER   = "#CBD5E1"       # Slate 300
C_PRIMARY  = "#1E40AF"       # Blue 800
C_ACCENT   = "#0284C7"       # Sky 600
C_EDGE_HW  = "#047857"       # Emerald 700
C_ML       = "#7C3AED"       # Violet 700
C_APP      = "#B45309"       # Amber 700
C_TEXT     = "#1E293B"       # Slate 800
C_MUTED    = "#64748B"       # Slate 500
C_LINE     = "#475569"       # Slate 600
C_BOX_FILL = "#F1F5F9"       # Slate 100

def draw_rounded_card(x, y, w, h, title, subtitle, fill_color, border_color, title_color=C_PRIMARY):
    # Background card
    card = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.6,rounding_size=1.2",
        linewidth=1.5,
        edgecolor=border_color,
        facecolor=fill_color,
        zorder=1
    )
    ax.add_patch(card)
    
    # Header bar
    header_h = 4.2
    header = patches.FancyBboxPatch(
        (x, y + h - header_h), w, header_h,
        boxstyle="round,pad=0.4,rounding_size=0.8",
        linewidth=0,
        facecolor=title_color,
        zorder=2
    )
    ax.add_patch(header)
    
    # Title text
    ax.text(x + w / 2, y + h - header_h / 2, title,
            ha="center", va="center", color="#FFFFFF",
            fontsize=11, fontweight="bold", zorder=3)
    
    if subtitle:
        ax.text(x + w / 2, y + h - header_h - 1.5, subtitle,
                ha="center", va="center", color=C_MUTED,
                fontsize=8.5, fontstyle="italic", zorder=3)

def draw_sub_box(x, y, w, h, title, items, tag="", fill_bg="#FFFFFF", edge_c=C_BORDER):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.4,rounding_size=0.8",
        linewidth=1.2,
        edgecolor=edge_c,
        facecolor=fill_bg,
        zorder=2
    )
    ax.add_patch(box)
    
    # Box title
    ax.text(x + 1.2, y + h - 1.8, title,
            ha="left", va="center", color=C_TEXT,
            fontsize=9.2, fontweight="bold", zorder=3)
    
    if tag:
        tag_box = patches.FancyBboxPatch(
            (x + w - len(tag)*1.05 - 1.2, y + h - 2.5), len(tag)*1.05, 1.6,
            boxstyle="round,pad=0.2,rounding_size=0.4",
            linewidth=0.5,
            edgecolor=C_MUTED,
            facecolor="#E2E8F0",
            zorder=3
        )
        ax.add_patch(tag_box)
        ax.text(x + w - (len(tag)*1.05)/2 - 1.2, y + h - 1.7, tag,
                ha="center", va="center", color="#334155",
                fontsize=7.2, fontweight="bold", zorder=4)

    # Bullet items
    line_y = y + h - 3.8
    for item in items:
        ax.text(x + 1.4, line_y, f"• {item}",
                ha="left", va="center", color="#334155",
                fontsize=8.0, zorder=3)
        line_y -= 1.8

def draw_flow_arrow(x1, y1, x2, y2, label=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_LINE,
            lw=1.8,
            shrinkA=4,
            shrinkB=4,
            mutation_scale=14,
        ),
        zorder=5
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 1.2, label, ha="center", va="bottom",
                color=C_PRIMARY, fontsize=7.8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec=C_BORDER, lw=0.8),
                zorder=6)

# ==============================================================================
# HEADER BANNER
# ==============================================================================
banner = patches.FancyBboxPatch(
    (2, 93), 96, 6.0,
    boxstyle="round,pad=0.5,rounding_size=0.8",
    linewidth=1.2,
    edgecolor=C_PRIMARY,
    facecolor=C_TITLE_BG,
    zorder=2
)
ax.add_patch(banner)

ax.text(50, 96.6, "END-TO-END CYBER-PHYSICAL TINYML SYSTEM ARCHITECTURE",
        ha="center", va="center", color="#FFFFFF", fontsize=15, fontweight="bold", zorder=3)
ax.text(50, 94.2, "Water Pump Fault Diagnostics: Multi-Modal Edge Sensing, Session-Isolated ML Training & Embedded TinyML Inference",
        ha="center", va="center", color="#94A3B8", fontsize=9.2, zorder=3)

# ==============================================================================
# TIER 1: PHYSICAL TEST RIG & MULTI-MODAL TRANSDUCERS (Left Column: X: 2 to 24)
# ==============================================================================
draw_rounded_card(2, 4, 21, 87, "1. SENSING & PHYSICAL ASSET", "Physical Testbed & Sensor Array", "#FFFFFF", C_BORDER, title_color="#0F766E")

draw_sub_box(
    3.5, 68, 18, 17.5,
    "Physical Pump Asset",
    [
        "Centrifugal/Submersible pump",
        "Variable water flow load",
        "Continuous 24/7 hydraulic cycle",
        "Controlled fault test bench:",
        " - Normal circulating load",
        " - Dry run (cavitation)",
        " - Overload (throttling)",
        " - Overheating (thermal stress)",
        " - Bearing fault (vibration)",
    ],
    tag="PLANT", fill_bg="#F0FDFA", edge_c="#99F6E4"
)

draw_sub_box(
    3.5, 47, 18, 19,
    "MPU-6050 6-Axis IMU",
    [
        "Inter-Integrated Circuit (I2C)",
        "SDA -> GPIO 21 | SCL -> GPIO 22",
        "I2C Address: 0x68 (Standard)",
        "Scale: ±2g full-scale range",
        "Sensitivity: 16,384 LSB/g",
        "3-Axis Acceleration (ax, ay, az)",
        "Fixed pump casing mount",
    ],
    tag="VIBRATION", fill_bg="#F8FAFC", edge_c="#CBD5E1"
)

draw_sub_box(
    3.5, 26, 18, 19,
    "ACS712-20A Current Sensor",
    [
        "Hall-effect isolated sensing",
        "Analog Output -> GPIO 34 (ADC1)",
        "Voltage Divider: 10kΩ / 20kΩ",
        "Scaling Ratio: 2/3 (3.3V safe)",
        "Sensitivity: 100 mV/A (0.100 V/A)",
        "Reconstructed Zero: 2.364 V",
        "Oversampled motor current",
    ],
    tag="CURRENT", fill_bg="#F8FAFC", edge_c="#CBD5E1"
)

draw_sub_box(
    3.5, 6, 18, 18,
    "DS18B20 Digital Temp",
    [
        "Dallas 1-Wire Digital Bus",
        "Data Pin -> GPIO 4",
        "4.7 kΩ External Pull-up",
        "10-Bit ADC Resolution",
        "Conversion Time: 187.5 ms",
        "Physical Range: -20°C to +125°C",
        "Motor stator/casing surface",
    ],
    tag="TEMP", fill_bg="#F8FAFC", edge_c="#CBD5E1"
)

# ==============================================================================
# TIER 2: EDGE EMBEDDED COMPUTING & DSP (Column 2: X: 26 to 48)
# ==============================================================================
draw_rounded_card(26, 4, 22, 87, "2. EDGE MCU & EMBEDDED DSP", "ESP32 Cyber-Physical Signal Pipeline", "#FFFFFF", C_BORDER, title_color=C_EDGE_HW)

draw_sub_box(
    27.5, 73.5, 19, 12,
    "ESP32-WROOM-32",
    [
        "Dual-Core Xtensa LX6 @ 240 MHz",
        "520 KB SRAM | 4 MB Flash",
        "Built-in 12-bit SAR ADC",
        "Hardware I2C + 1-Wire + UART",
        "Non-blocking FreeRTOS timers",
    ],
    tag="DUAL CORE", fill_bg="#ECFDF5", edge_c="#A7F3D0"
)

draw_sub_box(
    27.5, 49.5, 19, 22,
    "On-Chip Feature Extraction",
    [
        "Vibration Sampling Engine:",
        " • 50 samples @ 5ms (250ms win)",
        " • Magnitude: mag = sqrt(x²+y²+z²)",
        " • RMS: vib_rms = sqrt(Σmag²/50)",
        " • Peak: vib_peak = max(mag)",
        "Current Oversampling & Recon:",
        " • 30 ADC samples filtered",
        " • pin_V = (avgADC / 4095) * 3.3V",
        " • V_sensor = pin_V / (2/3 ratio)",
        " • I = |V_sensor - 2.364| / 0.100",
        "Asynchronous Temp Parsing (°C)",
    ],
    tag="DSP ENGINE", fill_bg="#F0FDF4", edge_c="#BBF7D0"
)

draw_sub_box(
    27.5, 27.5, 19, 20,
    "Telemetry Frame Emission",
    [
        "High-Speed UART (115,200 baud)",
        "Exact 9-Column CSV Framing:",
        " [timestamp, vib_rms, vib_peak,",
        "  vib_x, vib_y, vib_z, current,",
        "  temperature, label]",
        "Session State Handshake:",
        " • '0' to '4' Command trigger",
        " • 3-Minute duration controller",
        " • # SESSION COMPLETE signaling",
    ],
    tag="PROTOCOL", fill_bg="#F8FAFC", edge_c="#CBD5E1"
)

draw_sub_box(
    27.5, 6, 19, 19.5,
    "Embedded TinyML Inference",
    [
        "TensorFlow Lite for Microcontrollers",
        "Pre-allocated Tensor Arena (6 KB)",
        "Zero dynamic RAM allocation",
        "Input Feature Preprocessing:",
        " • norm[i] = (x[i] - μ[i]) / σ[i]",
        "Inference Latency: < 4.2 ms",
        "Local Fault State Output (0 to 3)",
        "Dual Mode: Logging vs Autonomous",
    ],
    tag="TINYML RUNTIME", fill_bg="#EFF6FF", edge_c="#BFDBFE"
)

# ==============================================================================
# TIER 3: DATA INGESTION & DATASET PROTOCOL (Column 3: X: 51 to 73)
# ==============================================================================
draw_rounded_card(51, 4, 22, 87, "3. DATA INGESTION & PROVENANCE", "Serial Client & Session Isolation", "#FFFFFF", C_BORDER, title_color=C_PRIMARY)

draw_sub_box(
    52.5, 73.5, 19, 12,
    "Python Serial Client",
    [
        "serial_logger.py client",
        "Auto-reconnect on COM dropouts",
        "Checksum & frame-level validation",
        "Strict physical bound assertions",
        "Real-time streaming feedback",
    ],
    tag="ACQUISITION", fill_bg="#F0F9FF", edge_c="#BAE6FD"
)

draw_sub_box(
    52.5, 51.5, 19, 20,
    "Session-Based Storage",
    [
        "Independent recording sessions:",
        " • normal_session_1.csv (Test)",
        " • normal_session_2.csv (Train)",
        " • dryrun_session_1.csv (Test)",
        " • dryrun_session_2.csv (Train)",
        " • overheat_session_1.csv (Test)",
        " • overheat_session_2.csv (Train)",
        " • fault_session_1/2.csv (Train/Test)",
        "Eliminates row-level leakage",
    ],
    tag="ISOLATION", fill_bg="#F8FAFC", edge_c="#CBD5E1"
)

draw_sub_box(
    52.5, 30.0, 19, 19.5,
    "Provenance & Audit Tracking",
    [
        "dataset_manifest.json catalog",
        "Hardware origin verification",
        "Explicit tag: 'measured' vs 'unknown'",
        "File-level checksum and row tally",
        "Label distribution verification:",
        " • df['label'].value_counts()",
        " • Imbalance ratio detection",
    ],
    tag="AUDIT TRAIL", fill_bg="#F8FAFC", edge_c="#CBD5E1"
)

draw_sub_box(
    52.5, 6, 19, 22,
    "Cross-Validation Protocol",
    [
        "Hold-Out Test Split: session_1",
        "Development Split: session_2 & 3",
        "Leave-One-Session-Out CV (LOSO)",
        "StandardScaler fitted strictly",
        "on development sessions:",
        " • Mean (μ) and StdDev (σ) frozen",
        " • Applied identically to Test Set",
        "Prevents cross-session overfitting",
    ],
    tag="VALIDATION", fill_bg="#EFF6FF", edge_c="#BFDBFE"
)

# ==============================================================================
# TIER 4: CLOUD / RESEARCH ML BENCHMARK & TINYML EXPORT (Column 4: X: 76 to 98)
# ==============================================================================
draw_rounded_card(76, 4, 22, 87, "4. ML BENCHMARK & DEPLOYMENT", "Model Architecture, Tuning & Output", "#FFFFFF", C_BORDER, title_color=C_ML)

draw_sub_box(
    77.5, 73.5, 19, 12,
    "Multi-Model Benchmarking",
    [
        "1. Logistic Regression (Linear baseline)",
        "2. Random Forest (300 estimators)",
        "   - Permutation feature importance",
        "   - Sensor ablation diagnostics",
        "3. Deep MLP Classifier (TinyML)",
    ],
    tag="EVALUATION", fill_bg="#FAF5FF", edge_c="#E9D5FF"
)

draw_sub_box(
    77.5, 49.5, 19, 22,
    "Target Neural Network (MLP)",
    [
        "Layer 1: Input (7 sensor features)",
        "Layer 2: Dense (32 units, ReLU)",
        "Layer 3: Dropout (rate = 0.15)",
        "Layer 4: Dense (16 units, ReLU)",
        "Layer 5: Dense (4 units, Softmax)",
        "Optimization & Regularization:",
        " • Loss: Sparse Categorical Cross-Entropy",
        " • Optimizer: Adam (lr = 0.001)",
        " • Session-level early stopping",
    ],
    tag="TINY MLP", fill_bg="#F5F3FF", edge_c="#DDD6FE"
)

draw_sub_box(
    77.5, 28.5, 19, 19,
    "Model Quantization & Export",
    [
        "TFLite FlatBuffer Conversion:",
        " • Size: 5.69 KB (Ultra-compact)",
        "C++ Header Array Export:",
        " • pump_condition_model_data.h",
        "Numerical Parity Check:",
        " • Keras vs TFLite probability diff:",
        "   max_abs_err < 1e-7",
        " • Class prediction agreement: 100.0%",
    ],
    tag="TFLITE EXPORT", fill_bg="#FDF4FF", edge_c="#F5D0FE"
)

draw_sub_box(
    77.5, 6, 19, 20.5,
    "Digital Twin & Telemetry",
    [
        "MQTT Broker (Eclipse Mosquitto)",
        "Topic: pump/telemetry & pump/state",
        "Digital Twin Web Dashboard:",
        " • Real-time 3D vibration/current",
        " • Multi-class fault probability gauge",
        " • Health degradation score (0-100%)",
        " • Instant predictive trip alerts",
    ],
    tag="DIGITAL TWIN", fill_bg="#FEF3C7", edge_c="#FDE68A"
)

# ==============================================================================
# PIPELINE DATA FLOW ARROWS (INTER-TIER CONNECTIONS)
# ==============================================================================
# Tier 1 to Tier 2 (Sensors -> MCU)
draw_flow_arrow(23, 56.5, 26, 60.5, "I2C (400kHz)")
draw_flow_arrow(23, 35.5, 26, 55.0, "ADC1 (12b)")
draw_flow_arrow(23, 15.0, 26, 50.0, "1-Wire")

# Tier 2 to Tier 3 (MCU -> Python Ingestion)
draw_flow_arrow(48, 37.5, 51, 61.5, "UART CSV Stream")

# Tier 3 to Tier 4 (Ingestion/Data -> ML Model Training)
draw_flow_arrow(73, 61.5, 76, 60.5, "StandardScaler")

# Tier 4 to Tier 2 (TinyML C++ Header Model Deployment back to ESP32!)
ax.annotate(
    "", xy=(48, 15.7), xytext=(76, 38.0),
    arrowprops=dict(
        arrowstyle="-|>",
        color="#7C3AED",
        lw=2.0,
        ls="--",
        shrinkA=4,
        shrinkB=4,
        connectionstyle="arc3,rad=-0.15",
        mutation_scale=14,
    ),
    zorder=6
)
ax.text(62, 23.5, "Embedded TFLite Model Export\n(pump_condition_model_data.h)",
        ha="center", va="center", color="#7C3AED", fontsize=8.2, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FAF5FF", ec="#DDD6FE", lw=1.0),
        zorder=7)

# Tier 2 to Tier 4 (MQTT Live Telemetry to Digital Twin)
ax.annotate(
    "", xy=(76, 16.0), xytext=(48, 11.5),
    arrowprops=dict(
        arrowstyle="-|>",
        color="#D97706",
        lw=2.0,
        ls="--",
        shrinkA=4,
        shrinkB=4,
        connectionstyle="arc3,rad=0.12",
        mutation_scale=14,
    ),
    zorder=6
)
ax.text(62, 10.5, "Live MQTT Telemetry & Health States\n(WiFi / 115200 Baud)",
        ha="center", va="center", color="#B45309", fontsize=8.2, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FEF3C7", ec="#FDE68A", lw=1.0),
        zorder=7)

# Bottom Footer Annotation
footer_text = "Figure: Comprehensive Cyber-Physical TinyML Architecture. Depicting physical transducer integration, edge DSP feature extraction, session-isolated validation protocol, neural network training, TFLite compression, and digital twin monitoring."
ax.text(50, 1.2, footer_text, ha="center", va="center", color=C_MUTED, fontsize=8.2, fontstyle="italic")

# Save high-resolution PNG and vector SVG
png_path = OUTPUT_DIR / "00_system_architecture_diagram.png"
svg_path = OUTPUT_DIR / "00_system_architecture_diagram.svg"

plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
plt.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
plt.close(fig)

print(f"[OK] High-resolution architecture diagram saved to:\n  - {png_path}\n  - {svg_path}")
