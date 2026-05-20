import numpy as np
import matplotlib.pyplot as plt
import requests
import io

def calc_es(Tc):
    """
    Calculates the saturation vapor pressure (es) over liquid water.
    
    Parameters:
    Tc (float or ndarray): Temperature in degrees Celsius.

    Returns:
    es (float or ndarray): Saturation vapor pressure in hectopascals (hPa) 
                           or millibars (mb).
    """
    #es = 6.1094 * np.exp(17.625 * Tc / (243.04 + Tc))
    es = 6.112 * np.exp(17.67*Tc/(243.5+Tc))
    return es

def svinvert(T0, q0, p0, p):
    """
    Calculates temperature and saturation specific humidity along a 
    reversible adiabat.
    
    Parameters:
    T0 : float - Initial parcel temperature (K)
    q0 : float - Initial specific humidity (kg/kg)
    p0 : float - Initial pressure (hPa)
    p  : array_like - Vector of pressures to lift to (hPa)
    
    Returns:
    T  : ndarray - Temperatures (K)
    qs : ndarray - Saturation specific humidities (kg/kg)
    """
    # Constants
    cp = 1005.0
    cpv = 1870.0
    Rd = 287.0
    Cl = 4160.0
    Rv = 461.0
    Lv0 = 2.5e6
    pref = 1000.0
    Tref = 290.0
    tolerance = 0.002
    
    # Vapor pressure, saturation vapor pressure, and initial entropy (s0)
    e0 = p0 * q0 / (0.622 + q0)
    Tc = T0 - 273.15
    es = calc_es(Tc)
    Lv = Lv0 + (cpv - Cl) * Tc
    
    # Calculate initial entropy
    # Assumes there is no initial condensed water
    s0 = (cp + Cl * q0) * np.log(T0 / Tref) - Rd * np.log((p0 - e0) / pref) + \
         (Lv * q0 / T0) - (q0 * Rv * np.log(e0 / es))
    
    # Ensure p is a numpy array
    p = np.atleast_1d(p)
    psize = p.size
    T = np.zeros(psize)
    qs = np.zeros(psize)
    
    # First guess for temperature
    Tg = 280.0
    
    for i in range(psize):
        delta = 10.0
        while delta > tolerance:
            Tgc = Tg - 273.15
            Lv_current = Lv0 + (cpv - Cl) * Tgc
            
            # Parcel vapor pressure if it were unsaturated at this p
            e00 = p[i] * q0 / (0.622 + q0)
            
            # Saturation vapor pressure at current guess Tg
            esg = calc_es(Tgc)
            qsg = 0.622 * esg / (p[i] - esg)
            
            # Logic for undersaturation vs saturation
            e_actual = min(e00, esg)
            q_actual = min(qsg, q0)
            
            # Invert entropy equation for T
            term1 = s0 + Rd * np.log((p[i] - e_actual) / pref)
            term2 = - (Lv_current * q_actual / Tg)
            term3 = q_actual * Rv * np.log(e_actual / esg)
            
            T_new = Tref * np.exp((term1 + term2 + term3) / (cp + Cl * q0))
            
            delta = abs(T_new - Tg)
            # Relaxation factor for convergence
            Tg = 0.65 * Tg + 0.35 * T_new
            
        T[i] = Tg
        Tgc = Tg - 273.15
        es_final = calc_es(Tgc)
        qs[i] = 0.622 * es_final / (p[i] - es_final)
        
    return T, qs

def svinvertpa(T0, q0, p0, p):
    """
    Calculates temperature and saturation specific humidity along a 
    pseudo-adiabat based on Bryan (2008).
    
    Parameters:
    T0 : float - Initial parcel temperature (K)
    q0 : float - Initial specific humidity (kg/kg)
    p0 : float - Initial pressure (hPa)
    p  : array_like - Vector of pressures to lift to (hPa)
    
    Returns:
    T  : ndarray - Temperatures (K)
    qs : ndarray - Saturation specific humidities (kg/kg)
    """
    # Constants
    cp = 1005.0
    Rd = 287.0
    Rv = 461.0
    Lv = 2.555e6  # Constant latent heat value per Bryan (2008)
    pref = 1000.0
    Tref = 290.0
    tolerance = 0.002
    
    # Vapor pressure and initial entropy (s0)
    e0 = p0 * q0 / (0.622 + q0)
    Tc0 = T0 - 273.15
    # Initial saturation vapor pressure for entropy calculation
    es0 = calc_es(Tc0)
    
    # Initial entropy (Dry air partial pressure used)
    s0 = cp * np.log(T0 / Tref) - Rd * np.log((p0 - e0) / pref) + \
         (Lv * q0 / T0) - (q0 * Rv * np.log(e0 / es0))
    
    p = np.atleast_1d(p)
    psize = p.size
    T = np.zeros(psize)
    qs = np.zeros(psize)
    
    # Initial guess for the iterative solver
    Tg = 280.0
    
    for i in range(psize):
        delta = 10.0
        while delta > tolerance:
            Tgc = Tg - 273.15
            
            # Saturation vapor pressure at current guess Tg
            esg = calc_es(Tgc)
            qsg = 0.622 * esg / (p[i] - esg)
            
            # Vapor pressure if unsaturated at this pressure
            e00 = p[i] * q0 / (0.622 + q0)
            
            # Use current conditions or saturation conditions
            e_effective = min(e00, esg)
            q_effective = min(qsg, q0)
            
            # Invert entropy equation for T (Pseudo-adiabatic uses only cp)
            term1 = s0 + Rd * np.log((p[i] - e_effective) / pref)
            term2 = - (Lv * q_effective / Tg)
            term3 = q_effective * Rv * np.log(e_effective / esg)
            
            T_new = Tref * np.exp((term1 + term2 + term3) / cp)
            
            delta = abs(T_new - Tg)
            # Relaxation to ensure convergence
            Tg = 0.65 * Tg + 0.35 * T_new
            
        T[i] = Tg
        Tgc = Tg - 273.15
        es_final = calc_es(Tgc)
        qs[i] = 0.622 * es_final / (p[i] - es_final)
        
    return T, qs

def getsounding(station_id, year, month, day, hour):
    """
    Retrieves a rawinsonde sounding from the University of Wyoming database.
    """
    station_id = str(station_id)
    header = [
        'P (hPa)', 'z (m)', 'T (C)', 'DWPT (C)', 'RELH (%)', 
        'MIXR (g/kg)', 'DRCT (deg)', 'SKNT (kts)', 'THTA (K)', 
        'THTE (K)', 'THTV (K)'
    ]
    
    # Format date/time strings with leading zeros
    year_str = str(year)
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)
    hour_str = str(hour).zfill(2)
    
    # Construct the URL
    theurl = (
        f"http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&"
        f"TYPE=TEXT:LIST&YEAR={year_str}&MONTH={month_str}&"
        f"FROM={day_str}{hour_str}&TO={day_str}{hour_str}&STNM={station_id}"
    )
    
    try:
        response = requests.get(theurl)
        # status_code 200 is success
        if response.status_code != 200:
            print("Error: URL retrieval failed")
            return np.nan, header, 0
        
        text = response.text
        
        # Check if the returned page is too short (implies an error or no data)
        if len(text) < 900:
            return np.nan, header, 0
        
        # Locate data within <PRE> tags
        start_tag = '<PRE>'
        end_tag = '</PRE>'
        
        if start_tag not in text:
            print(" ")
            return 0, header, 0
            
        kstart = text.find(start_tag)
        kend = text.find(end_tag)
        
        # Extract the table content
        # The offset (318) in the original Matlab code skips the headers inside the <PRE> tag
        data_block = text[kstart + 318 : kend - 2]
        
        # Read the fixed-width text block into a list of numbers
        raw_data = []
        # Using StringIO to treat the string like a file
        f = io.StringIO(data_block)
        
        for line in f:
            if len(line.strip()) == 0:
                continue
            
            # Helper logic similar to str2n (fixed-width parsing)
            # Column slices based on the original Matlab code indices
            slices = [
                (0, 7), (8, 14), (15, 21), (21, 28), (29, 35), 
                (36, 42), (44, 49), (51, 56), (57, 63), (64, 70), (71, 77)
            ]
            
            row = []
            for start, end in slices:
                val_str = line[start:end].strip()
                try:
                    row.append(float(val_str))
                except ValueError:
                    row.append(np.nan)
            
            if len(row) == 11:
                raw_data.append(row)
        
        data = np.array(raw_data)
        return data, header, 1
        
    except Exception as e:
        print(f"Error: {e}")
        return np.nan, header, 0

def skewtv(pz, tz, rhz):
    """
    Calculates and displays a Skew-T diagram as a function of virtual temperature.
    
    Parameters:
    pz  : array_like - Pressure (hPa)
    tz  : array_like - Temperature (C)
    rhz : array_like - Relative humidity (0-1)
    """
    # 1. Calculate Dewpoint Temperature (tdz)
    ez = calc_es(tz)
    qz = rhz * 0.622 * ez / (pz - ez)
    # chi is related to vapor pressure for dewpoint calculation
    chi = np.log(pz * qz / (6.1094 * (0.622 + qz)))
    tdz = 243.04 * chi / (17.625 - chi)
    
    # 2. Setup Background Grid
    p = np.arange(1050, 70, -25)
    t0_axis = np.arange(-48, 52, 2)
    
    Lvcp = 2.555e3 / 1005.0
    Rdcp = 287.0 / 1005.0
    
    # Initialize background field arrays
    thet = np.zeros((len(t0_axis), len(p)))
    thetaea = np.zeros((len(t0_axis), len(p)))
    q_grid = np.zeros((len(t0_axis), len(p)))
    tvem = np.zeros((len(t0_axis), len(p)))

    # 3. Calculate Background Isopleths (Dry/Moist Adiabats and Mixing Ratio)
    for i, t_val in enumerate(t0_axis):
        for j, p_val in enumerate(p):
            # tv is the skewed temperature coordinate
            tv = t_val + 30.0 * np.log(0.001 * p_val)
            qg = 0.0 # Initial guess for mixing ratio
            
            # Iterate to find Temperature (TK) given Virtual Temperature (tv)
            for n in range(3):
                TK = (273.15 + tv) * (1 + qg) / (1 + qg / 0.622)
                Tc = TK - 273.15
                es = calc_es(Tc)
                qg = 0.622 * es / (p_val - es)
            
            q_grid[i, j] = 1000 * qg # g/kg
            # Potential Temperature
            theta_val = TK * (1000.0 / p_val)**Rdcp
            # Equivalent Potential Temperature (approx)
            thetaea[i, j] = theta_val * np.exp(Lvcp * q_grid[i, j] / TK)
            # Virtual Potential Temperature
            thet[i, j] = theta_val * (1 + qg / 0.622) / (1 + qg)
            tvem[i, j] = tv

    # 4. Create the Plot
    plt.figure(figsize=(8, 6))
    
    # Background Skewed Isotherms (Black)
    plt.contour(t0_axis, p, tvem.T, 16, colors='k', linewidths=0.5)
    
    # Dry Adiabats (Blue)
    plt.contour(t0_axis, p, thet.T, 24, colors='b', linewidths=0.5, alpha=0.6)
    
    # Saturation Mixing Ratio lines (Green) - plotting sqrt(q) as per original
    plt.contour(t0_axis, p, np.sqrt(q_grid.T), 24, colors='g', linewidths=0.5, alpha=0.6)
    
    # Pseudo-adiabats / Theta-e (Red)
    plt.contour(t0_axis, p, thetaea.T, 24, colors='r', linewidths=0.5, alpha=0.6)
    
    # 5. Plot the actual sounding data
    # Convert sounding T and Td to Skew-T coordinates
    tvz = (tz + 273.15) * (1 + qz / 0.622) / (1 + qz) - 273.15
    tzm = tvz - 30.0 * np.log(0.001 * pz)
    tdzm = tdz - 30.0 * np.log(0.001 * pz)
    
    plt.plot(tzm, pz, 'r', linewidth=2, label='Virtual Temp')
    plt.plot(tdzm, pz, 'g', linewidth=2, label='Dewpoint')
    
    # 6. Formatting
    plt.yscale('log')
    plt.gca().invert_yaxis()
    plt.ylim(pz[0], 70)
    plt.xlim(-40, 40)
    
    levels = [70, 100, 200, 300, 400, 500, 700, 850, 1000]
    plt.yticks(levels, [str(l) for l in levels])
    
    # Use 'major' to only show grid lines where the yticks are
    plt.grid(True, which='major', axis='y', color='gray', linestyle='-', alpha=0.3)
    plt.gca().yaxis.set_minor_formatter(plt.NullFormatter())

    plt.xlabel('Density Temperature (C)', fontweight='bold')
    plt.ylabel('Pressure (hPa)', fontweight='bold')
    plt.legend(loc='best')

def get_station_id(search_key):
    search_key = search_key.lower().strip()
    matches = []
    filename = 'helpers/stations.txt'
    with open(filename, 'r') as f:
        for line in f:
            if ',' in line:
                s_id, s_name = line.strip().split(',', 1)
                s_id = s_id.strip()
                s_name = s_name.strip()
                    
                # Requirement: search_key must be in the station name
                if search_key in s_name.lower():
                    matches.append((s_id, s_name))

    # Handle the results
    if not matches:
        print(f"No station found containing '{search_key}'.")
        return None
    
    if len(matches) == 1:
        # Only one match found, return it directly
        return matches[0][0]
    else:
        # Multiple matches found (e.g., searching "Oslo" finds two stations)
        print(f"\nMultiple matches found for '{search_key}':")
        for i, (sid, name) in enumerate(matches):
            print(f"{i+1}) ID: {sid} | Name: {name}")
            
        choice = input("\nWhich one do you want? (Enter number): ")
        try:
            selection = int(choice) - 1
            return matches[selection][0]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return None