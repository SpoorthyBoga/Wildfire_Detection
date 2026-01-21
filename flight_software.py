import numpy as np
import struct
from math import log

# Landsat Constants
K1 = 774.8853
K2 = 1321.0789

class RadiationHardenedSoftware:
    """
    Implements Software-Level Triple Modular Redundancy (TMR)
    and CCSDS Packet Construction.
    Ref: Section III.C of your IEEE Paper.
    """
    
    def radiance_to_temp(self, L):
        if L <= 0: return 0.0
        return K2 / log((K1 / L) + 1)

    def _trigger_logic(self, b7, b10):
        """
        Stage 1: Multi-Spectral Wake-Up Trigger
        Returns: (Trigger_Boolean, Confidence_Estimate, Temperature)
        """
        T = self.radiance_to_temp(b10.max())
        swir_max = b7.max()
        
        # Logic: High Temp + High SWIR (Fire) vs High Temp + Low SWIR (Sun Glint)
        if T < 290: return False, 0.0, T
        if 290 <= T < 310: return False, 0.3, T
        
        # Fire Confirmation Threshold
        if T >= 310 and swir_max > 0.25:
            conf = min((T - 300) / 80, 1.0)
            return True, conf, T
            
        return False, 0.0, T

    def protected_execution(self, b7, b10, inject_fault=False):
        """
        Executes trigger logic 3 times (TMR) to protect against bit-flips.
        """
        results = []
        
        # Run 1
        results.append(self._trigger_logic(b7, b10))
        # Run 2
        res2 = self._trigger_logic(b7, b10)
        if inject_fault: 
            # Simulate a cosmic ray flipping a boolean
            res2 = (not res2[0], res2[1], res2[2])
        results.append(res2)
        # Run 3
        results.append(self._trigger_logic(b7, b10))

        # Majority Vote
        votes = [r[0] for r in results]
        final_trigger = 1 if sum(votes) >= 2 else 0
        
        # Telemetry Data (Average of valid runs)
        avg_conf = sum([r[1] for r in results]) / 3
        avg_temp = sum([r[2] for r in results]) / 3
        
        # Detect if correction happened
        fault_detected = not (results[0][0] == results[1][0] == results[2][0])
            
        return bool(final_trigger), avg_conf, avg_temp, fault_detected

def construct_ccsds_packet(lat, lon, conf, frp, timestamp):
    """
    Constructs a 26-byte binary payload (Table I in paper).
    """
    # Header (Mock 6 bytes)
    header = b'\x1A\xCF\xFC\x1D\x00\x00'
    # GPS Time (Double - 8 bytes)
    t_bytes = struct.pack('>d', timestamp)
    # Lat/Lon (Int - 4 bytes each, scaled)
    lat_b = struct.pack('>i', int(lat * 1e6))
    lon_b = struct.pack('>i', int(lon * 1e6))
    # Conf (Byte - 1 byte)
    conf_b = struct.pack('>B', int(conf * 100))
    # FRP (Short - 2 bytes)
    frp_b = struct.pack('>H', int(frp))
    
    return header + t_bytes + lat_b + lon_b + conf_b + frp_b