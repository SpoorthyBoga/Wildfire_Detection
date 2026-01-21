import time
import numpy as np
import csv
import random
from flight_software import RadiationHardenedSoftware, construct_ccsds_packet
from inference_engine import EarlyExitModel

# --- SETUP ---
obc = RadiationHardenedSoftware()
ai_core = EarlyExitModel("student_model.npy")
rows = list(csv.DictReader(open("landsat8_dataset/labels.csv")))

# Select interesting test cases
test_indices = [0, 1500, 2999] # Background, Early Fire, Active Fire
scenarios = ["SAFE_TERRAIN", "EARLY_IGNITION", "CATASTRROPHIC_FIRE"]

print("\n" + "█"*60)
print("   SATELLITE ONBOARD PROCESSING UNIT (SDME ARCHITECTURE)")
print("   Status: ORBITAL OPS | Mode: AUTONOMOUS")
print("█"*60 + "\n")

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
        
        ai_conf, exit_stage, latency = ai_core.infer(b7, b10)
        
        print(f"      -> Model: MobileNetV2-Quantized")
        print(f"      -> Pathway: {exit_stage}")
        print(f"      -> Compute Latency: {latency} ms")
        print(f"      -> Fire Probability: {ai_conf:.2%}")

        # --- STEP 3: DOWNLINK ---
        if ai_conf > 0.5:
            print("  [3] Constructing Alert Packet...")
            pkt = construct_ccsds_packet(
                lat=17.385, lon=78.486, 
                conf=ai_conf, frp=temp*1.2, timestamp=time.time()
            )
            print(f"      -> PROTOCOL: CCSDS Space Packet")
            print(f"      -> PAYLOAD: {pkt.hex().upper()[:40]}... (Size: {len(pkt)} bytes)")
            print(f"      -> STATUS: Queued for S-Band Downlink.")
    else:
        print("  [2] Trigger NEGATIVE. Discarding Data to save Power.")

print("\n" + "█"*60)
print("✔ DEMO SEQUENCE COMPLETE")