import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import csv
import time
import psutil
import os
import platform
from flight_software import RadiationHardenedSoftware, construct_ccsds_packet
from inference_engine import EarlyExitModel

# --- CONFIGURATION ---
# "Cyberpunk" Style Colors
STYLE = {
    'bg': '#050505',       # Pitch Black
    'text': '#00FF41',     # Hacker Green
    'warm': '#FF3333',     # Alert Red
    'cool': '#00EAFF',     # Cyan
    'dim': '#444444',      # Gray
    'warn': '#FFD700'      # Gold
}

class MissionControl:
    def __init__(self):
        # Initialize Backend Systems
        self.obc = RadiationHardenedSoftware()
        self.ai = EarlyExitModel("student_model.npy")
        self.rows = list(csv.DictReader(open("landsat8_dataset/labels.csv")))
        self.scenarios = [0, 1500, 2999] # Safe, Early, Active
        self.current_idx = 0
        self.log_buffer = ["INITIALIZING KERNEL...", "LOADING TMR MODULES...", "SYSTEM READY."]

        # --- SETUP PLOT UI ---
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title('SDME: SATELLITE ONBOARD AI')
        
        # Create a 3x3 Grid
        gs = gridspec.GridSpec(3, 3, figure=self.fig, height_ratios=[1, 1, 1])
        
        # 1. MAIN SENSOR FEED (Left Panel - Large)
        self.ax_map = self.fig.add_subplot(gs[:, :2])
        self.ax_map.axis('off')
        
        # 2. LIVE TELEMETRY (Top Right)
        self.ax_tel = self.fig.add_subplot(gs[0, 2])
        self.ax_tel.set_title(" SENSOR TELEMETRY ", color=STYLE['cool'], fontsize=10, weight='bold')
        self.ax_tel.axis('off')

        # 3. HARDWARE MONITOR (Middle Right - NEW!)
        self.ax_hw = self.fig.add_subplot(gs[1, 2])
        self.ax_hw.set_title(" HARDWARE MANIFEST ", color=STYLE['text'], fontsize=10, weight='bold')
        self.ax_hw.axis('off')

        # 4. KERNEL LOG (Bottom Right)
        self.ax_log = self.fig.add_subplot(gs[2, 2])
        self.ax_log.set_title(" OBC SYSTEM LOG ", color=STYLE['dim'], fontsize=8)
        self.ax_log.axis('off')

        # Instructions Footer
        self.fig.text(0.02, 0.02, "CONTROLS: [SPACE] Next Scene | [F] Inject Radiation Fault", 
                     color=STYLE['dim'], fontsize=10)

        # Connect Keypress
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Load Initial Scene
        self.load_scene()

    def get_ram_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024) # MB

    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        self.log_buffer.append(f"[{timestamp}] {msg}")
        if len(self.log_buffer) > 12: self.log_buffer.pop(0)

    def draw_telemetry(self, temp, conf, is_fire, fault):
        self.ax_tel.clear()
        self.ax_tel.axis('off')
        self.ax_tel.set_xlim(0, 1)
        self.ax_tel.set_ylim(0, 4)
        self.ax_tel.set_title(" SENSOR TELEMETRY ", color=STYLE['cool'], fontsize=10, weight='bold')

        # Helper to draw bars
        def draw_bar(y, label, val, max_val, color):
            norm = min(val / max_val, 1.0)
            self.ax_tel.text(0, y+0.4, f"{label}: {val:.1f}", color='white', fontsize=9)
            self.ax_tel.barh(y, norm, color=color, height=0.2, align='center')
            self.ax_tel.barh(y, 1.0, color='#222222', height=0.2, zorder=-1, align='center')

        # Temp Bar
        c_temp = STYLE['warm'] if temp > 310 else STYLE['cool']
        draw_bar(3, "PEAK TEMP (K)", temp, 400, c_temp)
        
        # Confidence Bar
        c_conf = STYLE['text'] if conf > 0.5 else STYLE['dim']
        draw_bar(2, "AI CONFIDENCE", conf, 1.0, c_conf)
        
        # Status
        status = "CRITICAL" if is_fire else "NOMINAL"
        s_col = STYLE['warm'] if is_fire else STYLE['text']
        self.ax_tel.text(0, 0.8, "MISSION STATUS:", color='gray', fontsize=10)
        self.ax_tel.text(0.5, 0.8, status, color=s_col, fontsize=14, weight='bold')

        if fault:
            self.ax_tel.text(0, 0.2, "⚠ SEU DETECTED (CORRECTED)", color=STYLE['warn'], fontsize=9, weight='bold')

    def draw_hardware(self):
        """Draws the RAM usage and Hardware Specs Comparison"""
        self.ax_hw.clear()
        self.ax_hw.axis('off')
        self.ax_hw.set_title(" HARDWARE MANIFEST ", color=STYLE['text'], fontsize=10, weight='bold')
        self.ax_hw.set_xlim(0, 1)
        self.ax_hw.set_ylim(0, 5)

        # Specs Text
        self.ax_hw.text(0, 4.5, "TARGET: LEON3 (SPARC V8)", color='white', fontsize=9, weight='bold')
        self.ax_hw.text(0, 4.0, "LIMIT:  512 MB / 500 MHz", color='gray', fontsize=9)
        
        self.ax_hw.text(0, 3.0, f"HOST:   {platform.system().upper()} (DEMO)", color='white', fontsize=9, weight='bold')
        
        # RAM Monitor Bar
        current_ram = self.get_ram_usage()
        limit_ram = 512.0
        
        self.ax_hw.text(0, 1.5, f"ACTIVE RAM USAGE: {current_ram:.1f} MB", color=STYLE['cool'], fontsize=10)
        
        # Draw RAM Bar
        norm = min(current_ram / limit_ram, 1.0)
        col = STYLE['text'] if current_ram < 100 else STYLE['warn']
        self.ax_hw.barh(0.8, norm, color=col, height=0.3)
        self.ax_hw.barh(0.8, 1.0, color='#222222', height=0.3, zorder=-1)
        
        self.ax_hw.text(0, 0.2, "EFFICIENCY: OPTIMAL", color=STYLE['text'], fontsize=8)

    def draw_log(self):
        self.ax_log.clear()
        self.ax_log.axis('off')
        self.ax_log.set_title(" OBC KERNEL LOG ", color=STYLE['dim'], fontsize=8)
        
        for i, line in enumerate(reversed(self.log_buffer)):
            col = 'white'
            if "ALERT" in line: col = STYLE['warn']
            if "PACKET" in line: col = STYLE['cool']
            if "CRITICAL" in line: col = STYLE['warm']
            self.ax_log.text(0, i * 0.12, line, color=col, fontsize=9, family='monospace')
        self.ax_log.set_ylim(-0.05, 1.0)

    def load_scene(self):
        idx = self.scenarios[self.current_idx]
        sid = self.rows[idx]["sample_id"]
        
        self.b7 = np.load(f"landsat8_dataset/band7_swir/{sid}.npy")
        self.b10 = np.load(f"landsat8_dataset/band10_thermal/{sid}.npy")
        
        # Display Image
        self.ax_map.clear()
        self.ax_map.imshow(self.b10, cmap='inferno')
        self.ax_map.set_title(f"SENSOR FEED: {sid}", color=STYLE['cool'], fontsize=14)
        self.ax_map.axis('off')
        
        self.log(f"SCENE ACQUIRED: {sid}")
        self.run_logic(inject_fault=False)

    def run_logic(self, inject_fault):
        # 1. TMR Trigger
        is_fire, trig_conf, temp, fault = self.obc.protected_execution(
            self.b7, self.b10, inject_fault=inject_fault
        )
        
        if inject_fault:
            self.log("!!! RADIATION STRIKE DETECTED !!!")
            self.log(">> TMR: Bit-Flip Corrected")
        
        final_conf = 0.0
        
        # 2. AI Inference
        if is_fire:
            final_conf, exit_mode, latency = self.ai.infer(self.b7, self.b10)
            self.log(f"AI WAKEUP: {exit_mode}")
            
            # Draw Target Box
            y, x = np.unravel_index(np.argmax(self.b10), self.b10.shape)
            rect = Rectangle((x-2, y-2), 5, 5, fill=False, edgecolor=STYLE['warm'], linewidth=2)
            self.ax_map.add_patch(rect)
            self.ax_map.text(x+3, y, "TARGET LOCKED", color=STYLE['warm'], fontsize=8)

            if final_conf > 0.5:
                pkt = construct_ccsds_packet(17.0, 78.0, final_conf, temp, time.time())
                self.log(f"TX PACKET: {pkt.hex()[:15]}...")
        else:
            self.log("TRIGGER NEGATIVE. SLEEP MODE.")

        # 3. Update All Panels
        self.draw_telemetry(temp, final_conf, is_fire, fault or inject_fault)
        self.draw_hardware()
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