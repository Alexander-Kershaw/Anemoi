from __future__ import annotations 

import numpy as np
import pandas as pd

# This module contains meteorological calculation functions used in Notus

# Meteorological calculation functions (derived variables)

def wind_spd_kmh_to_ms(wspd_kmh: pd.Series) -> pd.Series:
    
    # Convert wind speed from kilometers per hour to meters per second

    return wspd_kmh * (1000.0 / 3600.0)


def saturation_vapor_pressure(temp_c: pd.Series) -> pd.Series:
    
    # Calculate saturation vapor pressure over liquid water using the Magnus formula

    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    

def vapor_pressure_from_dewpoint_hpa(dewpoint_c: pd.Series) -> pd.Series:
    
    # Calculate actual vapor pressure from dew point temperature

    """
    Dewpoint is the temperature at which air becomes saturated with moisture.
    Therefore actual vapor pressure can be calculated as the saturation vapor pressure
    """

    return saturation_vapor_pressure(dewpoint_c)

def specific_humidity_g_per_kg(dewpoint_c: pd.Series, pres_hpa: pd.Series) -> pd.Series:

    # Calculate specific humidity (g/kg) from dewpoint and pressure

    """
    e = vapor pressure (hPa)
    Mixing radio w = 0.622 * e / (p - e)
    Specific humidity q = w / (1 + w) * 1000 (to convert to g/kg)

    Note: requires pressure, othersise skip this variable
    """

    e = vapor_pressure_from_dewpoint_hpa(dewpoint_c)
    w = 0.622 * e / (pres_hpa - e)
    q = w / (1 + w) * 1000.0 
    
    return q


def wind_vec_components_ms(wspd_ms: pd.Series, wdir_deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Compute wind vector components (u, v) in m/s from speed and meteorological direction

    Meteorological convention:
    - wdir is the direction the wind is COMING FROM (0°=North, 90°=East).
    Components are defined as:
    u = -V * sin(theta)
    v = -V * cos(theta)
    """
    theta = np.deg2rad(wdir_deg)
    u = -wspd_ms * np.sin(theta)
    v = -wspd_ms * np.cos(theta)
    return u, v


def dewpoint_from_temp_rh_c(temp_c: pd.Series, rh_percent: pd.Series) -> pd.Series:
    """
    Approximate dew point (°C) from temperature (°C) and relative humidity (%)

    Utilizes a Magnus-type formula:
      gamma = ln(RH/100) + (a*T)/(b+T)
      Td = (b*gamma)/(a-gamma)

    with a=17.625, b=243.04°C (common over water, good for typical surface temps).

    Notes:
    - This is an approximation, but widely used and very useful operationally.
    - RH must be in (0, 100]; we clip to avoid ln(0).
    """
    rh = rh_percent.clip(lower=0.1, upper=100.0) / 100.0 # Convert RH to fraction and avoid log(0)
    a = 17.625
    b = 243.04
    gamma = np.log(rh) + (a * temp_c) / (b + temp_c) 
    td = (b * gamma) / (a - gamma)
    return td



