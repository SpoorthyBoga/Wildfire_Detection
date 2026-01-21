# flight_software.py
# -------------------------------------------------
# Satellite Flight Software
# Physics-based Fire Trigger (HOTSPOT LOGIC)
# -------------------------------------------------

import numpy as np

MAX_DN = 65535.0
DEMO_MODE = True   # Demo configuration

def sanity_check(arr):
    if arr is None or arr.size == 0:
        return False
    return True

def normalize_dn(dn):
    return dn / MAX_DN

def brightness_temperature_proxy(band10_norm):
    return band10_norm * 330.0

def radiometric_fire_trigger(band7_dn, band10_dn):
    """
    Stage 1: Physics-based fire trigger
    DEMO MODE: calibrated for synthetic Landsat-like data
    """

    if band7_dn is None or band10_dn is None:
        return False, 0.0

    band7 = band7_dn / 65535.0
    band10 = band10_dn / 65535.0

    # Thermal proxy (Kelvin-like)
    temp_max = band10.max() * 330.0
    swir_max = band7.max()

    # ---------------- DEMO MODE ----------------
    if DEMO_MODE:
        # Background
        if temp_max < 285:
            return False, 0.0

        # Weak fire (early stage)
        if 285 <= temp_max < 300:
            return False, 0.2

        # Strong fire (active)
        if temp_max >= 300 and swir_max > 0.15:
            confidence_hint = min((temp_max - 295) / 40, 1.0)
            return True, confidence_hint

    # ---------------- FLIGHT MODE ----------------
    if temp_max >= 320 and swir_max > 0.25:
        confidence_hint = min((temp_max - 310) / 50, 1.0)
        return True, confidence_hint

    return False, 0.0

