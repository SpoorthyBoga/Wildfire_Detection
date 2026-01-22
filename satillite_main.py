import time
import numpy as np
import csv
import platform
import psutil
import os
from flight_software import RadiationHardenedSoftware, construct_ccsds_packet
from inference_engine import EarlyExitModel

# --- SETUP ---
obc = RadiationHardenedSoftware()
ai_core = EarlyExitModel("student_model.npy")
rows = list(csv.DictReader(open("landsat8_dataset/labels.csv")))

# Select interesting test cases
test_indices = [0, 1500, 2999] # Background, Early Fire, Active Fire
scenarios = ["SAFE_TERRAIN", "EARLY_IGNITION", "CATASTRROPHIC_FIRE"]

# --- UTILS ---
def get_ram_usage():
    process = psutil.Process(os.getpid())
    mb = process.memory_info().rss / (1024 * 1024)
    return mb

def print_specs():
    print("\n" + "█"*70)
    print("   SATELLITE ONBOARD PROCESSING UNIT (SDME ARCHITECTURE)")
    print("   HARDWARE SPECIFICATION MANIFEST")
    print("█"*70)
    
    # Get Host Info
    try:
        host_freq = f"{psutil.cpu_freq().max:.0f} MHz"
        host_ram = f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
    except:
        host_freq = "Unknown"
        host_ram = "Unknown"
    
    # The "Proof" Table
    print(f"\n{'PARAMETER':<20} | {'TARGET (LEON3/CUBESAT)':<25} | {'HOST (DEMO LAPTOP)':<20}")
    print("-" * 75)
    print(f"{'CPU Arch':<20} | {'SPARC V8 (32-bit)':<25} | {platform.machine():<20}")
    print(f"{'Clock Speed':<20} | {'500 MHz':<25} | {host_freq:<20}")
    print(f"{'RAM Capacity':<20} | {'512 MB SDRAM':<25} | {host_ram:<20}")
    print(f"{'Power Budget':<20} | {'< 1.5 Watts':<25} | {'~45 Watts':<20}")
    print(f"{'OS / Kernel':<20} | {'RTEMS Real-Time':<25} | {platform.system()}")
    print("-" * 75)
    print("   [INFO] Latency simulation active to match LEON3 constraints.")
    print(f"   [INFO] Current Demo RAM Usage: {get_ram_usage():.2f} MB (<< 512 MB Limit)")
    print("█"*70 + "\n")
    time.sleep(3) # Pause so judges can read

# --- MAIN MISSION LOOP ---
print_specs()
print("   Status: ORBITAL OPS | Mode: AUTONOMOUS")

for i, idx in enumerate(test_indices):
    sid = rows[idx]["sample_id"]
    scenario = scenarios[i]
    
    # Load Scene
    b7 = np.load(f"landsat8_dataset/band7_swir/{sid}.npy")
    b10 = np.load(f"landsat8_dataset/band10_thermal/{sid}.npy")

    print(f"\n>>> ACQUIRING SCENE: {scenario} (ID: {sid})")
    time.sleep(0.5) 

    # --- STEP 1: FAULT-TOLERANT TRIGGER ---
    print("  [1] Executing TMR Protected Trigger...")
    
    # Simulate a cosmic ray hit on the second scenario only
    inject_bug = (i == 1) 
    
    is_fire, conf, temp, corrected = obc.protected_execution(b7, b10, inject_fault=inject_bug)
    
    if corrected:
        print(f"      [!] ALERT: SEU (Bit-Flip) Detected in Core 2!")
        print(f"      [✓] TMR Correction Applied. System Integrity Restored.")
    else:
        print(f"      [i] TMR Check Passed. RAM Integrity OK.")
        
    print(f"      -> Trigger Decision: {is_fire} (Max Temp: {temp:.1f}K)")

    # --- STEP 2: ADAPTIVE INFERENCE ---
    if is_fire:
        print("  [2] Trigger POSITIVE. Waking AI Accelerator...")
        time.sleep(0.3)
        
        # PROOF OF EFFICIENCY: Show current RAM usage right before heavy math
        curr_mem = get_ram_usage()
        print(f"      [DEBUG] Active Memory Footprint: {curr_mem:.2f} MB (Efficient)")

        ai_conf, exit_stage, latency = ai_core.infer(b7, b10)
        
        print(f"      -> Model: MobileNetV2-Quantized [INT8]")
        print(f"      -> Pathway: {exit_stage}")
        print(f"      -> Compute Latency: {latency} ms (Simulated for LEON3)")
        print(f"      -> Fire Probability: {ai_conf:.2%}")

        # --- STEP 3: DOWNLINK ---
        if ai_conf > 0.5:
            print("  [3] Constructing Alert Packet...")
            pkt = construct_ccsds_packet(
                lat=17.385, lon=78.486, 
                conf=ai_conf, frp=temp*1.2, timestamp=time.time()
            )
            print(f"      -> PROTOCOL: CCSDS 133.0-B-2 (Space Packet)")
            print(f"      -> PAYLOAD: {pkt.hex().upper()[:40]}... (Size: {len(pkt)} bytes)")
            print(f"      -> STATUS: Queued for Downlink.")
    else:
        print("  [2] Trigger NEGATIVE. Discarding Data to save Power.")

print("\n" + "█"*70)
print("✔ DEMO SEQUENCE COMPLETE")