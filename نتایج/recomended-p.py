def analyze_pressure_range(inp_file, margin=5.0):
    """
    Run EPANET once and analyze node pressures.
    Args:
        inp_file (str): path to EPANET .inp file
        margin (float): safety margin (m) for recommended range
    Returns:
        dict with min_pressure, max_pressure, recommended_range
    """
    import wntr
    import numpy as np

    wn = wntr.network.WaterNetworkModel(inp_file)
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # Extract all junction pressures at first timestep
    pressures = results.node['pressure'].iloc[0].values.astype(float)

    min_p = float(np.min(pressures))
    max_p = float(np.max(pressures))

    # recommended range = inside min/max with margin
    rec_low = max(min_p, min_p + margin)
    rec_high = max_p - margin
    if rec_high <= rec_low:  # fallback if margins overlap
        rec_low, rec_high = min_p, max_p

    return {
        "min_pressure": round(min_p, 2),
        "max_pressure": round(max_p, 2),
        "recommended_range": (round(rec_low, 2), round(rec_high, 2))
    }


# مثال استفاده:
inp_file = r"networks\WDN-\benchmark\2PRV.inp"   # مسیر فایل .inp
info = analyze_pressure_range(inp_file)
print("Min Pressure:", info["min_pressure"])
print("Max Pressure:", info["max_pressure"])
print("Recommended Range:", info["recommended_range"])
