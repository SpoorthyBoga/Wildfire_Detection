import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import time
from flight_software import RadiationHardenedSoftware, construct_ccsds_packet
from inference_engine import EarlyExitModel

# --- CONFIGURATION ---
PAUSE_TIME = 3 # Seconds to admire each image
obc = RadiationHardenedSoftware()
ai_core = EarlyExitModel("student_model.npy")
rows = list(csv.DictReader(open("landsat8_dataset/labels.csv")))

# Select specific scenarios for the visual demo
scenarios = [
    (0, "SAFE TERRAIN", "Nominal"), 
    (1500, "EARLY IGNITION", "Radiation Event"), 
    (2999, "ACTIVE WILDFIRE", "Critical")
]

# Setup the Visualization Window
plt.style.use('dark_background')
fig, (ax_img, ax_info) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 0.6]})
fig.canvas.manager.set_window_title('SDME ONBOARD MISSION CONTROL')

def update_dashboard(sid, title, status_text, b7, b10, is_fire, temp, conf, packet, fault):
    ax_img.clear()
    ax_info.clear()
    
    # --- LEFT PANEL: SATELLITE IMAGERY ---
    # Normalize b10 (Thermal) for display
    img_disp = (b10 - b10.min()) / (b10.max() - b10.min())
    ax_img.imshow(img_disp, cmap='inferno')
    ax_img.set_title(f"SENSOR FEED: {sid}", fontsize=12, color='cyan')
    ax_img.axis('off')
    
    # Draw Red Box if Fire
    if is_fire:
        # Find the hottest spot for the box
        y, x = np.unravel_index(np.argmax(b10), b10.shape)
        rect = patches.Rectangle((x-2, y-2), 5, 5, linewidth=2, edgecolor='red', facecolor='none')
        ax_img.add_patch(rect)
        ax_img.text(x, y-4, f"TEMP: {temp:.0f}K", color='red', fontsize=10, weight='bold')

    # --- RIGHT PANEL: TELEMETRY DATA ---
    ax_info.axis('off')
    # Header
    ax_info.text(0.05, 0.95, "MISSION TELEMETRY", fontsize=18, weight='bold', color='white')
    ax_info.text(0.05, 0.92, "-"*40, color='gray')
    
    # Status Lines
    ax_info.text(0.05, 0.80, f"SCENE TYPE:  {title}", fontsize=12, color='yellow')
    
    # TMR Status
    tmr_color = 'red' if fault else 'lime'
    tmr_text = "ERROR DETECTED (CORRECTED)" if fault else "INTEGRITY NOMINAL"
    ax_info.text(0.05, 0.70, f"CPU STATUS:  {tmr_text}", fontsize=12, color=tmr_color, weight='bold')
    
    # AI Stats
    if is_fire:
        ax_info.text(0.05, 0.55, f"AI TRIGGER:  POSITIVE", fontsize=12, color='red')
        ax_info.text(0.05, 0.48, f"CONFIDENCE:  {conf:.2%}", fontsize=12, color='white')
        ax_info.text(0.05, 0.41, f"LATENCY:     98ms (Simulated)", fontsize=12, color='white')
        
        # Packet Dump
        ax_info.text(0.05, 0.25, "DOWNLINK PACKET (CCSDS):", fontsize=10, color='cyan')
        hex_code = packet.hex().upper()
        # Split hex for readability
        chunk1 = hex_code[:20]
        chunk2 = hex_code[20:]
        ax_info.text(0.05, 0.20, f"{chunk1}...", fontfamily='monospace', fontsize=10, color='gray')
    else:
        ax_info.text(0.05, 0.55, f"AI TRIGGER:  NEGATIVE", fontsize=12, color='lime')
        ax_info.text(0.05, 0.48, "ACTION:      SLEEP MODE", fontsize=12, color='gray')

    plt.draw()
    plt.pause(0.1)

# --- MAIN LOOP ---
print("Initializing Mission Control Visualization...")

for idx, title, subtitle in scenarios:
    sid = rows[idx]["sample_id"]
    b7 = np.load(f"landsat8_dataset/band7_swir/{sid}.npy")
    b10 = np.load(f"landsat8_dataset/band10_thermal/{sid}.npy")

    # 1. RUN LOGIC
    inject_fault = (idx == 1500) # Inject fault only for Early Ignition
    is_fire, conf_trig, temp, fault = obc.protected_execution(b7, b10, inject_fault=inject_fault)
    
    final_conf = 0.0
    packet = b''
    
    if is_fire:
        final_conf, _, _ = ai_core.infer(b7, b10)
        if final_conf > 0.5:
             packet = construct_ccsds_packet(17.0, 78.0, final_conf, temp, time.time())
    
    # 2. UPDATE DISPLAY
    update_dashboard(sid, title, subtitle, b7, b10, is_fire, temp, final_conf, packet, fault)
    
    # Wait for judge to read
    plt.pause(PAUSE_TIME)

print("Visualization Complete.")
plt.show()