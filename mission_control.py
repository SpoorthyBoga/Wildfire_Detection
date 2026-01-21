import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
import csv
import time
import random
from flight_software import RadiationHardenedSoftware, construct_ccsds_packet
from inference_engine import EarlyExitModel

# --- CONFIGURATION ---
# "Cyberpunk" Style Colors for the Summit
STYLE = {
    'bg': '#050505',       # Pitch Black
    'text': '#00FF41',     # Hacker Green
    'warm': '#FF3333',     # Alert Red
    'cool': '#00EAFF',     # Cyan
    'dim': '#444444'       # Gray
}

class MissionControl:
    def __init__(self):
        self.obc = RadiationHardenedSoftware()
        self.ai = EarlyExitModel("student_model.npy")
        
        # Load Data
        self.rows = list(csv.DictReader(open("landsat8_dataset/labels.csv")))
        self.scenarios = [0, 1500, 2999] # Safe, Early, Active
        self.current_idx = 0
        self.log_buffer = ["INITIALIZING SYSTEM...", "WAITING FOR PILOT INPUT..."]

        # Setup Plot
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.canvas.manager.set_window_title('SDME: SATELLITE ONBOARD AI')
        
        # Grid Layout (3x3)
        gs = gridspec.GridSpec(3, 3, figure=self.fig)
        
        # 1. Main Sensor View (Left 2/3rds)
        self.ax_map = self.fig.add_subplot(gs[:, :2])
        self.ax_map.axis('off')
        
        # 2. Telemetry Bars (Top Right)
        self.ax_tel = self.fig.add_subplot(gs[0, 2])
        self.ax_tel.set_title(" LIVE TELEMETRY ", color=STYLE['cool'], fontsize=10, weight='bold')
        self.ax_tel.axis('off')

        # 3. System Log (Bottom Right)
        self.ax_log = self.fig.add_subplot(gs[1:, 2])
        self.ax_log.axis('off')
        self.ax_log.set_title(" OBC KERNEL LOG ", color=STYLE['dim'], fontsize=8)

        # Interaction Instructions
        self.fig.text(0.02, 0.02, "CONTROLS: [SPACE] Next Scene | [F] Inject Radiation Fault", 
                     color=STYLE['dim'], fontsize=10)

        # Event Listeners
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.load_scene()

    def log(self, msg, color=None):
        """Add message to the scrolling log window"""
        timestamp = time.strftime("%H:%M:%S")
        prefix = f"[{timestamp}] "
        self.log_buffer.append(prefix + msg)
        if len(self.log_buffer) > 16: self.log_buffer.pop(0)

    def draw_telemetry(self, temp, conf, is_fire, fault):
        self.ax_tel.clear()
        self.ax_tel.axis('off')
        self.ax_tel.set_xlim(0, 1)
        self.ax_tel.set_ylim(0, 5)
        
        # Draw Bars
        def draw_bar(y, label, val, max_val, color):
            norm = min(val / max_val, 1.0)
            self.ax_tel.text(0, y+0.5, f"{label}: {val:.1f}", color='white', fontsize=9)
            self.ax_tel.barh(y, norm, color=color, height=0.3, align='center')
            self.ax_tel.barh(y, 1.0, color='#222222', height=0.3, zorder=-1, align='center')

        # Temp Bar (Max 400K)
        c_temp = STYLE['warm'] if temp > 310 else STYLE['cool']
        draw_bar(4, "PEAK TEMP (K)", temp, 400, c_temp)
        
        # Confidence Bar (Max 1.0)
        c_conf = STYLE['text'] if conf > 0.5 else STYLE['dim']
        draw_bar(3, "AI CONFIDENCE", conf, 1.0, c_conf)
        
        # Status Text
        status = "CRITICAL" if is_fire else "NOMINAL"
        s_col = STYLE['warm'] if is_fire else STYLE['text']
        self.ax_tel.text(0, 1.5, "MISSION STATUS:", color='gray', fontsize=10)
        self.ax_tel.text(0.5, 1.5, status, color=s_col, fontsize=14, weight='bold')

        # Fault Indicator
        if fault:
            self.ax_tel.text(0, 0.5, "⚠ SEU DETECTED", color='yellow', fontsize=12, weight='bold')
            self.ax_tel.text(0, 0.1, "TMR CORRECTION ACTIVE", color=STYLE['text'], fontsize=8)

    def draw_log(self):
        self.ax_log.clear()
        self.ax_log.axis('off')
        # Render text from bottom up
        for i, line in enumerate(reversed(self.log_buffer)):
            col = 'white'
            if "ALERT" in line: col = 'yellow'
            if "PACKET" in line: col = STYLE['cool']
            if "CRITICAL" in line: col = STYLE['warm']
            
            self.ax_log.text(0, i * 0.06, line, color=col, fontsize=9, family='monospace')
            
        self.ax_log.set_ylim(-0.05, 1.0)

    def load_scene(self):
        # Get current scene data
        idx = self.scenarios[self.current_idx]
        sid = self.rows[idx]["sample_id"]
        
        self.b7 = np.load(f"landsat8_dataset/band7_swir/{sid}.npy")
        self.b10 = np.load(f"landsat8_dataset/band10_thermal/{sid}.npy")
        
        # Display Image
        self.ax_map.clear()
        self.ax_map.imshow(self.b10, cmap='inferno')
        self.ax_map.set_title(f"SENSOR FEED: {sid}", color=STYLE['cool'], fontsize=14)
        self.ax_map.axis('off')
        
        self.log(f"Scene {sid} Acquired.")
        self.log("Waiting for trigger...", color='gray')
        self.run_logic(inject_fault=False)

    def run_logic(self, inject_fault):
        # 1. FLIGHT SOFTWARE EXECUTION
        is_fire, trig_conf, temp, fault = self.obc.protected_execution(
            self.b7, self.b10, inject_fault=inject_fault
        )
        
        if inject_fault:
            self.log("!!! RADIATION STRIKE ON CPU !!!")
            self.log(">> TMR: Voting Discrepancy Found")
            self.log(">> TMR: Error Corrected via Majority")
        
        # 2. AI INFERENCE (If triggered)
        final_conf = 0.0
        if is_fire:
            final_conf, exit_mode, latency = self.ai.infer(self.b7, self.b10)
            self.log(f"TRIGGER: {exit_mode}")
            
            # Draw Bounding Box
            y, x = np.unravel_index(np.argmax(self.b10), self.b10.shape)
            rect = plt.Rectangle((x-2, y-2), 5, 5, fill=False, edgecolor=STYLE['warm'], linewidth=2)
            self.ax_map.add_patch(rect)
            self.ax_map.text(x+3, y, "TARGET LOCKED", color=STYLE['warm'], fontsize=8)

            if final_conf > 0.5:
                pkt = construct_ccsds_packet(17.0, 78.0, final_conf, temp, time.time())
                self.log(f"PACKET: {pkt.hex()[:20]}...", color='cyan')
        else:
            self.log("TRIGGER: Negative (Sleep Mode)")

        # 3. UPDATE UI
        self.draw_telemetry(temp, final_conf, is_fire, fault or inject_fault)
        self.draw_log()
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == ' ':
            self.current_idx = (self.current_idx + 1) % len(self.scenarios)
            self.load_scene()
        elif event.key == 'f':
            self.run_logic(inject_fault=True)

if __name__ == "__main__":
    print("Launching Dashboard...")
    dash = MissionControl()
    plt.show()