"""
CA MUTCD 2026 Traffic Signal Warrant Analysis
Streamlit App — 26warrant_app.py

All thresholds sourced from:
    CA MUTCD 2026 Chapter 4C (January 18, 2026)
    Reference PDF: C:\\Users\\anaft\\Warrant\\References\\camutcd-2026-4c.pdf
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import NamedTuple

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
from streamlit_folium import st_folium

try:
    import pdfplumber
    _PDFPLUMBER_OK = True
except ImportError:
    _PDFPLUMBER_OK = False

try:
    import pypdf
    _PYPDF_OK = True
except ImportError:
    _PYPDF_OK = False

try:
    import plotly.io as _pio
    _KALEIDO_OK = True
except Exception:
    _KALEIDO_OK = False

logger = logging.getLogger(__name__)

# ==============================================================================
# SECTION 2: CONSTANTS — CA MUTCD 2026, Chapter 4C
# ==============================================================================

PDF_REFERENCE_PATH = Path(r"C:\Users\anaft\Warrant\References\camutcd-2026-4c.pdf")
PDF_FOOTER_CITE = (
    "Thresholds per CA MUTCD 2026, Chapter 4C  |  "
    "Reference: camutcd-2026-4c.pdf"
)

# ── Warrant 1 — Table 4C-1 (CA MUTCD 2026, p.1074) ───────────────────────────
# Key: (major_key, minor_key) where key = min(lanes, 2)
# Value: (maj_100, maj_80, maj_70, maj_56, min_100, min_80, min_70, min_56)
W1_COND_A: dict[tuple[int, int], tuple[int, ...]] = {
    (1, 1): (500, 400, 350, 280, 150, 120, 105,  84),
    (2, 1): (600, 480, 420, 336, 150, 120, 105,  84),
    (2, 2): (600, 480, 420, 336, 200, 160, 140, 112),
    (1, 2): (500, 400, 350, 280, 200, 160, 140, 112),
}
W1_COND_B: dict[tuple[int, int], tuple[int, ...]] = {
    (1, 1): (750, 600, 525, 420,  75,  60,  53,  42),
    (2, 1): (900, 720, 630, 504,  75,  60,  53,  42),
    (2, 2): (900, 720, 630, 504, 100,  80,  70,  56),
    (1, 2): (750, 600, 525, 420, 100,  80,  70,  56),
}
# Tuple index offsets for W1 tables
_I_MAJ = (0, 1, 2, 3)  # 100%, 80%, 70%, 56%
_I_MIN = (4, 5, 6, 7)

# ── Warrant 2 — Figure 4C-1 / 4C-2 (CA MUTCD 2026, pp.1057–1058) ────────────
# Piecewise-linear breakpoints: (major_vph, min_minor_vph)
# Urban Figure 4C-1 — key: (major_key, minor_key)
W2_CURVES_URBAN: dict[tuple[int, int], np.ndarray] = {
    (2, 2): np.array([(300, 400), (500, 250), (800, 150), (1100, 115), (1400, 115)]),
    (1, 2): np.array([(300, 320), (500, 200), (800, 130), (1000, 115), (1400, 115)]),
    (1, 1): np.array([(300, 200), (500, 140), (700, 90),  (900,  80), (1400,  80)]),
    (2, 1): np.array([(300, 160), (500, 110), (700, 85),  (900,  80), (1400,  80)]),
}
# Rural / 70% Factor Figure 4C-2 (floor: 1-lane→60, 2-lane→80)
W2_CURVES_70PCT: dict[tuple[int, int], np.ndarray] = {
    (2, 2): np.array([(200, 280), (350, 175), (560, 105), (770,  80), (1000,  80)]),
    (1, 2): np.array([(200, 224), (350, 140), (560,  91), (700,  80), (1000,  80)]),
    (1, 1): np.array([(200, 140), (350,  98), (490,  63), (630,  60), (1000,  60)]),
    (2, 1): np.array([(200, 112), (350,  77), (490,  63), (630,  60), (1000,  60)]),
}

# ── Warrant 3 — Figure 4C-3 / 4C-4 (CA MUTCD 2026, pp.1059–1060) ────────────
# Urban Figure 4C-3 (floor: 1-lane→100, 2-lane→150)
W3_CURVES_URBAN: dict[tuple[int, int], np.ndarray] = {
    (2, 2): np.array([(400, 500), (600, 350), (900, 200), (1200, 150), (1800, 150)]),
    (1, 2): np.array([(400, 400), (600, 280), (900, 175), (1200, 150), (1800, 150)]),
    (1, 1): np.array([(400, 300), (600, 200), (900, 130), (1200, 100), (1800, 100)]),
    (2, 1): np.array([(400, 240), (600, 165), (900, 115), (1200, 100), (1800, 100)]),
}
# Rural / 70% Factor Figure 4C-4 (floor: 1-lane→75, 2-lane→100)
W3_CURVES_70PCT: dict[tuple[int, int], np.ndarray] = {
    (2, 2): np.array([(300, 350), (420, 245), (630, 140), (840, 100), (1300, 100)]),
    (1, 2): np.array([(300, 280), (420, 196), (630, 122), (840, 100), (1300, 100)]),
    (1, 1): np.array([(300, 210), (420, 140), (630,  91), (840,  75), (1300,  75)]),
    (2, 1): np.array([(300, 168), (420, 116), (630,  88), (840,  75), (1300,  75)]),
}

# ── Warrant 4 — Figures 4C-5…4C-8 (CA MUTCD 2026, pp.1061–1064) ─────────────
# Pedestrian 4-hour Urban, Figure 4C-5 (floor: std→107, slow→53)
W4_4HR_URBAN_STD: np.ndarray = np.array(
    [(300, 300), (500, 200), (700, 107), (1400, 107)]
)
W4_4HR_URBAN_SLOW: np.ndarray = np.array(
    [(300, 150), (500, 100), (700,  53), (1400,  53)]
)
# Pedestrian peak hour Urban, Figure 4C-6 (floor: std→133, slow→66)
W4_PH_URBAN_STD: np.ndarray = np.array(
    [(300, 500), (500, 350), (800, 200), (1000, 133), (1800, 133)]
)
W4_PH_URBAN_SLOW: np.ndarray = np.array(
    [(300, 250), (500, 175), (800, 100), (1000,  66), (1800,  66)]
)
# Pedestrian 4-hour Rural/70%, Figure 4C-7 (floor: std→75, slow→37)
W4_4HR_70PCT_STD: np.ndarray = np.array(
    [(200, 210), (350, 140), (490,  75), (1000,  75)]
)
W4_4HR_70PCT_SLOW: np.ndarray = np.array(
    [(200, 104), (350,  70), (490,  37), (1000,  37)]
)
# Pedestrian peak hour Rural/70%, Figure 4C-8 (floor: std→93, slow→46)
W4_PH_70PCT_STD: np.ndarray = np.array(
    [(200, 350), (350, 245), (560, 140), (700,  93), (1200,  93)]
)
W4_PH_70PCT_SLOW: np.ndarray = np.array(
    [(200, 175), (350, 122), (560,  70), (700,  46), (1200,  46)]
)

# ── Warrant 7 — Crash Tables (CA MUTCD 2026, pp.1075–1076) ───────────────────
# Table 4C-3: 3-year urban crash thresholds
# Key: (major_key, minor_key); Value: (4-leg-total, 3-leg-total, 4-leg-FI, 3-leg-FI)
W7_CRASH_3YR_URBAN: dict[tuple[int, int], tuple[int, ...]] = {
    (1, 1): (6, 5, 4, 4),
    (2, 1): (6, 5, 4, 4),
    (2, 2): (6, 5, 4, 4),
    (1, 2): (6, 5, 4, 4),
}
# Table 4C-4: 1-year rural/70% crash thresholds
W7_CRASH_1YR_RURAL: dict[tuple[int, int], tuple[int, ...]] = {
    (1, 1): ( 4, 3, 3, 3),
    (2, 1): (10, 9, 6, 6),
    (2, 2): (10, 9, 6, 6),
    (1, 2): ( 4, 3, 3, 3),
}
# Table 4C-5: 3-year rural/70% crash thresholds
W7_CRASH_3YR_RURAL: dict[tuple[int, int], tuple[int, ...]] = {
    (1, 1): (4, 3, 3, 3),
    (2, 1): (6, 5, 4, 4),
    (2, 2): (6, 5, 4, 4),
    (1, 2): (4, 3, 3, 3),
}
# 5 correctable crashes threshold — Section 4C.08
W7_CORRECTABLE_THRESHOLD = 5

# ── Warrant 9 — Adjustment Factor Tables (CA MUTCD 2026, pp.1075–1076) ───────
# Table 4C-6: Adjustment factor for daily rail frequency
W9_RAIL_AF: dict[str, float] = {
    "1":       0.67,
    "2":       0.91,
    "3-5":     1.00,
    "6-8":     1.18,
    "9-11":    1.25,
    "12+":     1.33,
}
# Table 4C-7: Adjustment factor for % high-occupancy buses
W9_BUS_AF: dict[str, float] = {
    "0%":   1.00,
    "2%":   1.09,
    "4%":   1.19,
    "6%+":  1.32,
}
# Table 4C-8: Truck adjustment factor (simplified — D >= 70 ft column)
W9_TRUCK_AF_D70: dict[str, float] = {
    "0-2.5%":   0.50,
    "2.6-7.5%": 0.75,
    "7.6-12.5%":1.00,
    "12.6-17.5%":1.15,
    "17.6-22.5%":1.35,
    "22.6-27.5%":1.64,
    "27.5%+":   2.09,
}

# ── PROWAG / Accessibility Checklist Items ────────────────────────────────────
PROWAG_CHECKLIST: list[str] = [
    "Continuous accessible pedestrian route maintained throughout construction zone",
    "Temporary curb ramps provided at all pedestrian crossings (PROWAG §R304)",
    "Detectable warning surfaces (truncated domes) installed at curb ramp landings",
    "Minimum 4.0-ft (48-in) clear path width maintained on all temporary sidewalks",
    "Pedestrian detour signs posted per CA MUTCD Chapter 6D (TTC Zone Standards)",
    "Accessible Pedestrian Signals (APS) maintained or temporarily provided (CA MUTCD 4E.06)",
    "Adequate lighting maintained at all crossings and detour paths",
    "PROWAG §R205 compliant blended transitions where ramps are not feasible",
    "Temporary pedestrian facilities are firm, stable, and slip-resistant",
    "Overhead clearance ≥ 8.0 ft on all pedestrian paths through work zone",
    "School routes and transit stops remain accessible; if rerouted, signed accordingly",
]

# ── EADT Table — Average Traffic Estimate (CA MUTCD 2026 Figure 4C-103) ──────
# Key: (major_key, minor_key); Value: (urban_maj_vpd, urban_min_vpd, rural_maj_vpd, rural_min_vpd)
EADT_COND_A: dict[tuple[int, int], tuple[int, ...]] = {
    (1, 1): (8000, 2400, 5600, 1680),
    (2, 1): (9600, 2400, 6720, 1680),
    (2, 2): (9600, 3200, 6720, 2240),
    (1, 2): (8000, 3200, 5600, 2240),
}

# ── Default K-factor for ADT-to-hourly conversion ────────────────────────────
DEFAULT_K_FACTOR = 0.09


# ==============================================================================
# SECTION 3: DATACLASSES
# ==============================================================================

@dataclass
class ProjectInfo:
    name: str = "Intersection Name"
    location: str = ""
    county: str = ""
    route: str = ""
    district: str = ""
    analyst: str = ""
    checker: str = ""
    calc_date: str = field(default_factory=lambda: date.today().isoformat())
    check_date: str = ""
    lat: float = 34.0522
    lon: float = -118.2437


@dataclass
class LaneConfig:
    major_lanes: int = 2
    minor_lanes: int = 1

    @property
    def key(self) -> tuple[int, int]:
        return (min(self.major_lanes, 2), min(self.minor_lanes, 2))


@dataclass
class HourlyVolumes:
    hour: int          # 0–23
    major_vph: int = 0
    minor_vph: int = 0
    ped_vph: int = 0
    bike_vph: int = 0


@dataclass
class WarrantResult:
    warrant_num: int
    name: str
    satisfied: bool
    notes: list[str] = field(default_factory=list)
    detail_rows: list[dict] = field(default_factory=list)


@dataclass
class TMCData:
    raw_df: pd.DataFrame
    hourly_df: pd.DataFrame
    peak_1hr: int         # hour of peak hour
    peak_4hrs: list[int]  # 4 highest hours
    peak_8hrs: list[int]  # 8 highest hours
    phf: float = 0.0


@dataclass
class ADTData:
    raw_df: pd.DataFrame
    k_factor: float
    hourly_df: pd.DataFrame   # estimated hourly volumes


# ==============================================================================
# SECTION 4: PDF REFERENCE LOADING
# ==============================================================================

def load_camutcd_reference(pdf_path: str | Path) -> dict:
    """
    Load CA MUTCD 2026 Chapter 4C from PDF and return extracted text/table dict.
    Falls back gracefully if the PDF cannot be read.
    """
    path = Path(pdf_path)
    result: dict = {"status": "not_found", "pages": {}, "tables": {}}

    if not path.exists():
        logger.warning("CA MUTCD PDF not found at %s", path)
        return result

    if _PDFPLUMBER_OK:
        return _extract_with_pdfplumber(path, result)
    if _PYPDF_OK:
        return _extract_with_pypdf(path, result)

    result["status"] = "no_library"
    logger.warning("Neither pdfplumber nor pypdf is available")
    return result


def _extract_with_pdfplumber(path: Path, result: dict) -> dict:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            result["total_pages"] = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                result["pages"][i + 1] = text[:2000]
            result["status"] = "loaded"
    except Exception as exc:
        logger.error("pdfplumber error: %s", exc)
        result["status"] = "error"
    return result


def _extract_with_pypdf(path: Path, result: dict) -> dict:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        result["total_pages"] = len(reader.pages)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            result["pages"][i + 1] = text[:2000]
        result["status"] = "loaded"
    except Exception as exc:
        logger.error("pypdf error: %s", exc)
        result["status"] = "error"
    return result


# ==============================================================================
# SECTION 5: DATA PARSING FUNCTIONS
# ==============================================================================

def parse_adt_csv(file_obj, k_factor: float = DEFAULT_K_FACTOR) -> ADTData:
    """Parse ADT CSV (Timestamp, Direction, Volume) and estimate hourly volumes."""
    df = pd.read_csv(file_obj)
    df.columns = [c.strip() for c in df.columns]
    hourly = _estimate_adt_hourly(df, k_factor)
    return ADTData(raw_df=df, k_factor=k_factor, hourly_df=hourly)


def _estimate_adt_hourly(df: pd.DataFrame, k_factor: float) -> pd.DataFrame:
    """Apply K-factor to daily volumes to estimate hourly distribution."""
    if "Direction" in df.columns and "Volume" in df.columns:
        total_adt = df["Volume"].sum()
    else:
        total_adt = df.iloc[:, -1].sum()
    peak_hour_vol = total_adt * k_factor
    hours = list(range(24))
    # Approximate diurnal distribution (typical urban shape)
    weights = [
        0.5, 0.3, 0.2, 0.2, 0.3, 0.6,
        1.5, 3.0, 4.5, 3.5, 3.0, 4.0,
        4.5, 3.5, 3.0, 3.5, 4.5, 5.0,
        4.0, 3.5, 3.0, 2.5, 1.5, 1.0,
    ]
    w_sum = sum(weights)
    vols = [round(total_adt * w / w_sum) for w in weights]
    return pd.DataFrame({"Hour": hours, "EstimatedVPH": vols, "PeakFactor": [v / (peak_hour_vol or 1) for v in vols]})


def parse_tmc_csv(file_obj) -> TMCData:
    """Parse TMC CSV (Timestamp, Approach, Movement, Volume, Ped, Bike) and detect peaks."""
    df = pd.read_csv(file_obj)
    df.columns = [c.strip() for c in df.columns]
    hourly = aggregate_tmc_hourly(df)
    peak_1 = detect_peak_1hr(hourly)
    peak_4 = detect_peak_nhrs(hourly, 4)
    peak_8 = detect_peak_nhrs(hourly, 8)
    phf = _compute_phf_from_raw(df, peak_1)
    return TMCData(
        raw_df=df, hourly_df=hourly,
        peak_1hr=peak_1, peak_4hrs=peak_4, peak_8hrs=peak_8, phf=phf,
    )


def aggregate_tmc_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 15-min TMC data into hourly totals by approach."""
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%H:%M", errors="coerce")
    df["Hour"] = df["Timestamp"].dt.hour
    grp = df.groupby("Hour").agg(
        TotalVehicles=("Volume", "sum"),
        TotalPed=("Ped", "sum"),
        TotalBike=("Bike", "sum"),
    ).reset_index()
    return grp


def detect_peak_1hr(hourly_df: pd.DataFrame) -> int:
    """Return the hour (0–23) with the highest total vehicle volume."""
    if hourly_df.empty:
        return 7
    idx = hourly_df["TotalVehicles"].idxmax()
    return int(hourly_df.loc[idx, "Hour"])


def detect_peak_nhrs(hourly_df: pd.DataFrame, n: int) -> list[int]:
    """Return the n hours (0–23) with the highest total vehicle volumes."""
    if hourly_df.empty:
        return list(range(n))
    top = hourly_df.nlargest(n, "TotalVehicles")
    return sorted(int(h) for h in top["Hour"].tolist())


def _compute_phf_from_raw(df: pd.DataFrame, peak_hour: int) -> float:
    """Compute peak-hour factor from 15-min intervals in peak hour."""
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%H:%M", errors="coerce")
    df["Hour"] = df["Timestamp"].dt.hour
    peak_data = df[df["Hour"] == peak_hour]
    if peak_data.empty:
        return 0.90
    qtr_totals = (
        peak_data.groupby("Timestamp")["Volume"].sum().sort_index()
    )
    if qtr_totals.empty:
        return 0.90
    total_hour = qtr_totals.sum()
    max_15min = qtr_totals.max()
    if max_15min == 0:
        return 0.90
    return round(total_hour / (4 * max_15min), 3)


def compute_phf(peak_hr_volume: int, max_15min_volume: int) -> float:
    """PHF = hourly volume / (4 × max 15-min count). CA MUTCD 2026 §4C.04."""
    if max_15min_volume == 0:
        return 0.0
    return round(peak_hr_volume / (4 * max_15min_volume), 3)


# ==============================================================================
# SECTION 6: WARRANT CALCULATION HELPERS
# ==============================================================================

def _lane_key(lanes: int) -> int:
    """Clamp lane count to table key (1 or 2)."""
    return min(lanes, 2)


def w1_threshold(lane_cfg: LaneConfig, major_idx: int, minor_idx: int) -> tuple[int, int]:
    """
    Return (major_threshold, minor_threshold) for Warrant 1 Condition A.
    Indices per _I_MAJ / _I_MIN constants.
    CA MUTCD 2026 Table 4C-1 (p.1074).
    """
    row_a = W1_COND_A[lane_cfg.key]
    return row_a[major_idx], row_a[minor_idx]


def w1_threshold_b(lane_cfg: LaneConfig, major_idx: int, minor_idx: int) -> tuple[int, int]:
    """Return (major_threshold, minor_threshold) for Warrant 1 Condition B."""
    row_b = W1_COND_B[lane_cfg.key]
    return row_b[major_idx], row_b[minor_idx]


def _interpolate_curve(pts: np.ndarray, major_vph: float) -> float:
    """Linearly interpolate minor-street threshold from curve breakpoints."""
    xs = pts[:, 0]
    ys = pts[:, 1]
    return float(np.interp(major_vph, xs, ys))


def w2_curve_check(major_vph: int, minor_vph: int, lane_cfg: LaneConfig, use_70pct: bool) -> bool:
    """
    Return True if the (major, minor) point plots ABOVE the W2 four-hour curve.
    CA MUTCD 2026 Figure 4C-1 (urban) / Figure 4C-2 (70% factor).
    """
    curves = W2_CURVES_70PCT if use_70pct else W2_CURVES_URBAN
    key = lane_cfg.key
    if key not in curves:
        key = (1, 1)
    threshold = _interpolate_curve(curves[key], major_vph)
    return minor_vph >= threshold


def w3_curve_check(major_vph: int, minor_vph: int, lane_cfg: LaneConfig, use_70pct: bool) -> bool:
    """
    Return True if (major, minor) plots ABOVE the W3 peak hour curve.
    CA MUTCD 2026 Figure 4C-3 (urban) / Figure 4C-4 (70% factor).
    """
    curves = W3_CURVES_70PCT if use_70pct else W3_CURVES_URBAN
    key = lane_cfg.key
    if key not in curves:
        key = (1, 1)
    threshold = _interpolate_curve(curves[key], major_vph)
    return minor_vph >= threshold


def w4_4hr_curve_check(major_vph: int, ped_vph: int, use_70pct: bool, slow_walkers: bool) -> bool:
    """
    Return True if pedestrian 4-hour volume satisfies Figure 4C-5 (or 4C-7).
    CA MUTCD 2026 Figures 4C-5, 4C-7.
    """
    if use_70pct:
        pts = W4_4HR_70PCT_SLOW if slow_walkers else W4_4HR_70PCT_STD
    else:
        pts = W4_4HR_URBAN_SLOW if slow_walkers else W4_4HR_URBAN_STD
    threshold = _interpolate_curve(pts, major_vph)
    return ped_vph >= threshold


def w4_peak_curve_check(major_vph: int, ped_vph: int, use_70pct: bool, slow_walkers: bool) -> bool:
    """
    Return True if pedestrian peak-hour volume satisfies Figure 4C-6 (or 4C-8).
    CA MUTCD 2026 Figures 4C-6, 4C-8.
    """
    if use_70pct:
        pts = W4_PH_70PCT_SLOW if slow_walkers else W4_PH_70PCT_STD
    else:
        pts = W4_PH_URBAN_SLOW if slow_walkers else W4_PH_URBAN_STD
    threshold = _interpolate_curve(pts, major_vph)
    return ped_vph >= threshold


def _use_70pct(speed_mph: float, pop_lt10k: bool) -> bool:
    """Return True when 70% factor column applies per CA MUTCD 2026 §4C.02.07."""
    return speed_mph > 40 or pop_lt10k


# ==============================================================================
# SECTION 7: WARRANT EVALUATION FUNCTIONS
# ==============================================================================

def eval_w1(
    hourly_vols: list[HourlyVolumes],
    lane_cfg: LaneConfig,
    speed_mph: float,
    pop_lt10k: bool,
    combo_trial_done: bool,
) -> WarrantResult:
    """
    Warrant 1 — Eight-Hour Vehicular Volume.
    CA MUTCD 2026 Section 4C.02, Table 4C-1 (p.1074).
    Evaluates Condition A, B, and Combination A+B.
    """
    use_70 = _use_70pct(speed_mph, pop_lt10k)
    maj_i = 2 if use_70 else 0   # 70% or 100% column
    min_i = maj_i + 4

    rows = []
    cond_a_met = cond_b_met = 0
    for v in hourly_vols:
        thr_a_maj, thr_a_min = w1_threshold(lane_cfg, maj_i, min_i)
        thr_b_maj, thr_b_min = w1_threshold_b(lane_cfg, maj_i, min_i)
        a_ok = v.major_vph >= thr_a_maj and v.minor_vph >= thr_a_min
        b_ok = v.major_vph >= thr_b_maj and v.minor_vph >= thr_b_min
        if a_ok:
            cond_a_met += 1
        if b_ok:
            cond_b_met += 1
        rows.append({
            "Hour": f"{v.hour:02d}:00",
            "Major VPH": v.major_vph,
            f"Maj Thr-A ({thr_a_maj})": "✓" if v.major_vph >= thr_a_maj else "✗",
            "Minor VPH": v.minor_vph,
            f"Min Thr-A ({thr_a_min})": "✓" if v.minor_vph >= thr_a_min else "✗",
            "Cond A": "✓" if a_ok else "✗",
            "Cond B": "✓" if b_ok else "✗",
        })

    sat_a = cond_a_met >= 8
    sat_b = cond_b_met >= 8
    # Combination A+B uses 80% (or 56%) columns
    combo_maj_i = 3 if use_70 else 1
    combo_min_i = combo_maj_i + 4
    combo_a = combo_b = 0
    for v in hourly_vols:
        ta_maj, ta_min = w1_threshold(lane_cfg, combo_maj_i, combo_min_i)
        tb_maj, tb_min = w1_threshold_b(lane_cfg, combo_maj_i, combo_min_i)
        if v.major_vph >= ta_maj and v.minor_vph >= ta_min:
            combo_a += 1
        if v.major_vph >= tb_maj and v.minor_vph >= tb_min:
            combo_b += 1
    sat_combo = combo_trial_done and combo_a >= 8 and combo_b >= 8

    satisfied = sat_a or sat_b or sat_combo
    notes = [
        f"Condition A: {cond_a_met}/8 hours met — {'SATISFIED' if sat_a else 'NOT SATISFIED'}",
        f"Condition B: {cond_b_met}/8 hours met — {'SATISFIED' if sat_b else 'NOT SATISFIED'}",
        f"Combination A+B (80%/{56 if use_70 else 80}% cols): {'SATISFIED' if sat_combo else 'NOT SATISFIED'}",
        f"70% factor applied: {use_70} (speed={speed_mph} mph, pop<10K={pop_lt10k})",
    ]
    return WarrantResult(1, "Eight-Hour Vehicular Volume", satisfied, notes, rows)


def eval_w2(
    four_hour_vols: list[HourlyVolumes],
    lane_cfg: LaneConfig,
    use_70pct: bool,
) -> WarrantResult:
    """
    Warrant 2 — Four-Hour Vehicular Volume.
    CA MUTCD 2026 Section 4C.03, Figures 4C-1 (urban) / 4C-2 (70%).
    All 4 plotted points must fall above the applicable curve.
    """
    rows = []
    passes = 0
    for v in four_hour_vols:
        above = w2_curve_check(v.major_vph, v.minor_vph, lane_cfg, use_70pct)
        if above:
            passes += 1
        curves = W2_CURVES_70PCT if use_70pct else W2_CURVES_URBAN
        key = lane_cfg.key if lane_cfg.key in curves else (1, 1)
        min_thresh = _interpolate_curve(curves[key], v.major_vph)
        rows.append({
            "Hour": f"{v.hour:02d}:00",
            "Major VPH": v.major_vph,
            "Minor VPH": v.minor_vph,
            "Curve Min": round(min_thresh),
            "Above Curve": "✓" if above else "✗",
        })
    satisfied = passes >= 4
    fig_ref = "Figure 4C-2 (70% factor)" if use_70pct else "Figure 4C-1 (Urban)"
    notes = [
        f"Points above curve: {passes}/4 required — {fig_ref}",
        "CA MUTCD 2026 Section 4C.03",
    ]
    return WarrantResult(2, "Four-Hour Vehicular Volume", satisfied, notes, rows)


def eval_w3(
    peak_vol: HourlyVolumes,
    lane_cfg: LaneConfig,
    use_70pct: bool,
    delay_veh_hrs: float,
    total_entering_vph: int,
    num_approaches: int,
) -> WarrantResult:
    """
    Warrant 3 — Peak Hour.
    CA MUTCD 2026 Section 4C.04, Figures 4C-3 / 4C-4.
    Part A: delay + volume + entering volume.
    Part B: graphical curve.
    """
    # Part A
    delay_thresh = 5 if lane_cfg.minor_lanes >= 2 else 4
    minor_thresh = 150 if lane_cfg.minor_lanes >= 2 else 100
    enter_thresh = 800 if num_approaches >= 4 else 650
    part_a = (
        delay_veh_hrs >= delay_thresh
        and peak_vol.minor_vph >= minor_thresh
        and total_entering_vph >= enter_thresh
    )
    # Part B
    part_b = w3_curve_check(peak_vol.major_vph, peak_vol.minor_vph, lane_cfg, use_70pct)
    satisfied = part_a or part_b
    fig_ref = "Figure 4C-4 (70%)" if use_70pct else "Figure 4C-3 (Urban)"
    notes = [
        f"Part A: delay≥{delay_thresh} veh-hr: {delay_veh_hrs}  |  "
        f"minor≥{minor_thresh} vph: {peak_vol.minor_vph}  |  "
        f"entering≥{enter_thresh}: {total_entering_vph}  →  {'SATISFIED' if part_a else 'NOT MET'}",
        f"Part B ({fig_ref}): major={peak_vol.major_vph}, minor={peak_vol.minor_vph} "
        f"→ {'ABOVE CURVE' if part_b else 'BELOW CURVE'}",
        "CA MUTCD 2026 Section 4C.04",
    ]
    return WarrantResult(3, "Peak Hour", satisfied, notes)


def eval_w4(
    four_hr_peds: list[tuple[int, int]],  # (major_vph, ped_vph) for 4 hours
    peak_hr_ped: tuple[int, int],         # (major_vph, ped_vph) for peak hour
    use_70pct: bool,
    slow_walkers: bool,
    dist_to_signal_ft: float,
    no_restrict_progression: bool,
) -> WarrantResult:
    """
    Warrant 4 — Pedestrian Volume.
    CA MUTCD 2026 Section 4C.05, Figures 4C-5 through 4C-8.
    Part 1A: 4-hour pedestrian curve; Part 1B: peak-hour pedestrian curve.
    Part 2: signal spacing condition.
    """
    part_1a = all(
        w4_4hr_curve_check(maj, ped, use_70pct, slow_walkers)
        for maj, ped in four_hr_peds
    )
    part_1b = w4_peak_curve_check(peak_hr_ped[0], peak_hr_ped[1], use_70pct, slow_walkers)
    part_2 = dist_to_signal_ft > 300 or no_restrict_progression
    satisfied = (part_1a or part_1b) and part_2
    rows = [
        {"Hour": i + 1, "Major VPH": maj, "Ped VPH": ped,
         "4-Hr Curve": "✓" if w4_4hr_curve_check(maj, ped, use_70pct, slow_walkers) else "✗"}
        for i, (maj, ped) in enumerate(four_hr_peds)
    ]
    notes = [
        f"Part 1A (4-hr curve {'4C-7' if use_70pct else '4C-5'}): {'SATISFIED' if part_1a else 'NOT MET'}",
        f"Part 1B (peak-hr curve {'4C-8' if use_70pct else '4C-6'}): {'SATISFIED' if part_1b else 'NOT MET'}",
        f"Part 2 (signal spacing >300 ft or no restriction): {'SATISFIED' if part_2 else 'NOT MET'}",
        f"Slow-walker factor (15th pct < 3.5 fps): {slow_walkers}",
    ]
    return WarrantResult(4, "Pedestrian Volume", satisfied, notes, rows)


def eval_w5(
    adequate_gaps_per_hr: int,
    children_per_hr: int,
    alternatives_tried: bool,
    dist_to_signal_ft: float,
    no_restrict_progression: bool,
) -> WarrantResult:
    """
    Warrant 5 — School Crossing.
    CA MUTCD 2026 Section 4C.06.
    Part A: inadequate gaps + student volume.
    Part B: signal spacing condition.
    """
    part_a_gaps = adequate_gaps_per_hr < children_per_hr
    part_a_children = children_per_hr > 20
    part_a = part_a_gaps and part_a_children and alternatives_tried
    part_b = dist_to_signal_ft > 300 or no_restrict_progression
    satisfied = part_a and part_b
    notes = [
        f"Part A: adequate gaps/hr ({adequate_gaps_per_hr}) < children/hr ({children_per_hr}): {part_a_gaps}",
        f"        children/hr > 20: {part_a_children}",
        f"        less-restrictive alternatives considered: {alternatives_tried}",
        f"Part B: signal spacing >300 ft or no restriction: {part_b}",
        "CA MUTCD 2026 Section 4C.06",
    ]
    return WarrantResult(5, "School Crossing", satisfied, notes)


def eval_w6(
    min_dist_ft: float,
    one_way_no_platoon: bool,
    two_way_provides_progression: bool,
) -> WarrantResult:
    """
    Warrant 6 — Coordinated Signal System.
    CA MUTCD 2026 Section 4C.07.
    All parts must be satisfied.
    """
    dist_ok = min_dist_ft > 1000
    platoon_ok = one_way_no_platoon or two_way_provides_progression
    satisfied = dist_ok and platoon_ok
    notes = [
        f"Distance to nearest signal > 1000 ft: {dist_ok} (actual {min_dist_ft:.0f} ft)",
        f"One-way no-platoon condition: {one_way_no_platoon}",
        f"Two-way provides progression: {two_way_provides_progression}",
        "CA MUTCD 2026 Section 4C.07",
    ]
    return WarrantResult(6, "Coordinated Signal System", satisfied, notes)


def eval_w7(
    correctable_crashes: int,
    crash_table_condition: bool,
    alternatives_tried: bool,
    w1_80pct_met: bool,
) -> WarrantResult:
    """
    Warrant 7 — Crash Experience.
    CA MUTCD 2026 Section 4C.08, Tables 4C-2 through 4C-5.
    """
    crashes_ok = correctable_crashes >= W7_CORRECTABLE_THRESHOLD
    satisfied = alternatives_tried and crashes_ok and crash_table_condition and w1_80pct_met
    notes = [
        f"Adequate trial of alternatives failed: {alternatives_tried}",
        f"Correctable crashes ≥ {W7_CORRECTABLE_THRESHOLD}/year: {correctable_crashes} → {crashes_ok}",
        f"Crash table condition met (Tables 4C-2/3/4/5): {crash_table_condition}",
        f"80% of W1 volumes met for 8 hours: {w1_80pct_met}",
        "CA MUTCD 2026 Section 4C.08",
    ]
    return WarrantResult(7, "Crash Experience", satisfied, notes)


def eval_w8(
    is_principal_route: bool,
    is_rural_suburban: bool,
    is_on_official_plan: bool,
    min_volumes_met: bool,
) -> WarrantResult:
    """
    Warrant 8 — Roadway Network.
    CA MUTCD 2026 Section 4C.09.
    """
    functional_ok = is_principal_route or is_rural_suburban or is_on_official_plan
    satisfied = functional_ok and min_volumes_met
    notes = [
        f"Principal through route: {is_principal_route}",
        f"Rural/suburban highway: {is_rural_suburban}",
        f"On official major route plan: {is_on_official_plan}",
        f"Minimum volume conditions met: {min_volumes_met}",
        "CA MUTCD 2026 Section 4C.09",
    ]
    return WarrantResult(8, "Roadway Network", satisfied, notes)


def _compute_w9_af(rail_key: str, bus_key: str, truck_key: str) -> float:
    """Compute combined adjustment factor for Warrant 9. Tables 4C-6/7/8."""
    af_rail = W9_RAIL_AF.get(rail_key, 1.0)
    af_bus = W9_BUS_AF.get(bus_key, 1.0)
    af_truck = W9_TRUCK_AF_D70.get(truck_key, 1.0)
    return af_rail * af_bus * af_truck


def eval_w9(
    track_dist_ft: float,
    major_vph: int,
    minor_raw_vph: int,
    rail_key: str,
    bus_key: str,
    truck_key: str,
    two_or_more_lanes_at_track: bool,
) -> WarrantResult:
    """
    Warrant 9 — Intersection Near a Grade Crossing.
    CA MUTCD 2026 Section 4C.10, Tables 4C-6/7/8, Figures 4C-9/10.
    Part A: track within 140 ft of stop line.
    Part B: volume above applicable curve after adjustment.
    """
    part_a = track_dist_ft <= 140
    af = _compute_w9_af(rail_key, bus_key, truck_key)
    adjusted_minor = minor_raw_vph * af
    # Simplified threshold per Figure 4C-9 / 4C-10 minimum (25 vph floor)
    # Full curve not digitized; use minimum threshold check
    min_vph_threshold = 25.0
    part_b = adjusted_minor >= min_vph_threshold and major_vph > 0
    satisfied = part_a and part_b
    notes = [
        f"Part A: track within 140 ft of stop line: {part_a} ({track_dist_ft:.0f} ft)",
        f"Adjustment factor (rail×bus×truck): {af:.3f}",
        f"Adjusted minor approach: {minor_raw_vph} × {af:.3f} = {adjusted_minor:.1f} vph",
        f"Part B: adjusted volume ≥ curve minimum: {part_b}",
        f"{'2+ lanes' if two_or_more_lanes_at_track else '1 lane'} at track — use Figure {'4C-10' if two_or_more_lanes_at_track else '4C-9'}",
        "CA MUTCD 2026 Section 4C.10",
    ]
    return WarrantResult(9, "Intersection Near Grade Crossing", satisfied, notes)


# ==============================================================================
# SECTION 8: GIS MAP RENDERING
# ==============================================================================

def render_folium_map(
    lat: float,
    lon: float,
    name: str,
    hourly_df: pd.DataFrame | None = None,
) -> folium.Map:
    """Build a Folium map with intersection marker and optional TMC heatmap."""
    m = folium.Map(location=[lat, lon], zoom_start=17)
    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Aerial",
    ).add_to(m)
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(name, max_width=300),
        tooltip=name,
        icon=folium.Icon(color="red", icon="road", prefix="fa"),
    ).add_to(m)
    if hourly_df is not None and not hourly_df.empty and "TotalVehicles" in hourly_df.columns:
        _add_volume_indicators(m, lat, lon, hourly_df)
    folium.LayerControl().add_to(m)
    return m


def _add_volume_indicators(
    m: folium.Map, lat: float, lon: float, hourly_df: pd.DataFrame
) -> None:
    """Add a circle marker scaled to peak volume."""
    peak_vol = int(hourly_df["TotalVehicles"].max())
    radius = min(peak_vol / 10, 200)
    folium.CircleMarker(
        [lat, lon],
        radius=radius,
        color="blue",
        fill=True,
        fill_opacity=0.2,
        tooltip=f"Peak volume: {peak_vol} vph",
    ).add_to(m)


# ==============================================================================
# SECTION 9: CHART GENERATION
# ==============================================================================

def _fig_to_png(fig: go.Figure) -> bytes:
    """Convert plotly figure to PNG bytes (requires kaleido)."""
    try:
        import plotly.io as pio
        return pio.to_image(fig, format="png", width=700, height=380, scale=1.5)
    except Exception:
        return b""


def make_w1_chart(hourly_vols: list[HourlyVolumes], lane_cfg: LaneConfig, use_70pct: bool) -> go.Figure:
    """Bar chart of hourly volumes vs W1 Condition A thresholds."""
    hours = [f"{v.hour:02d}:00" for v in hourly_vols]
    maj_idx = 2 if use_70pct else 0
    min_idx = maj_idx + 4
    thr_a = W1_COND_A[lane_cfg.key]
    maj_thr = thr_a[maj_idx]
    min_thr = thr_a[min_idx]
    fig = go.Figure()
    fig.add_bar(name="Major VPH", x=hours, y=[v.major_vph for v in hourly_vols], marker_color="steelblue")
    fig.add_bar(name="Minor VPH", x=hours, y=[v.minor_vph for v in hourly_vols], marker_color="salmon")
    fig.add_hline(y=maj_thr, line_dash="dash", line_color="steelblue", annotation_text=f"Maj Thr {maj_thr}")
    fig.add_hline(y=min_thr, line_dash="dash", line_color="salmon", annotation_text=f"Min Thr {min_thr}")
    fig.update_layout(
        title="Warrant 1 — 8-Hour Volumes vs Condition A Thresholds (CA MUTCD 2026 Table 4C-1)",
        barmode="group", xaxis_title="Hour", yaxis_title="Vehicles per Hour",
        legend=dict(x=0, y=1), template="plotly_white",
    )
    return fig


def make_w2_chart(four_hr_vols: list[HourlyVolumes], lane_cfg: LaneConfig, use_70pct: bool) -> go.Figure:
    """Scatter plot of four-hour volumes vs W2 curve."""
    curves = W2_CURVES_70PCT if use_70pct else W2_CURVES_URBAN
    key = lane_cfg.key if lane_cfg.key in curves else (1, 1)
    pts = curves[key]
    x_curve = np.linspace(pts[0, 0], pts[-1, 0], 100)
    y_curve = np.interp(x_curve, pts[:, 0], pts[:, 1])
    fig = go.Figure()
    fig.add_scatter(x=x_curve, y=y_curve, mode="lines", name="Warrant Curve", line_color="gray", line_dash="dash")
    colors = ["green" if w2_curve_check(v.major_vph, v.minor_vph, lane_cfg, use_70pct) else "red"
              for v in four_hr_vols]
    fig.add_scatter(
        x=[v.major_vph for v in four_hr_vols],
        y=[v.minor_vph for v in four_hr_vols],
        mode="markers+text",
        marker=dict(size=12, color=colors),
        text=[f"{v.hour:02d}:00" for v in four_hr_vols],
        textposition="top center",
        name="Study Hours",
    )
    fig_ref = "Fig 4C-2 (70%)" if use_70pct else "Fig 4C-1 (Urban)"
    fig.update_layout(
        title=f"Warrant 2 — Four-Hour Volume Plot vs {fig_ref} (CA MUTCD 2026)",
        xaxis_title="Major Street VPH (Both Approaches)",
        yaxis_title="Minor Street VPH (Higher Approach)",
        template="plotly_white",
    )
    return fig


def make_w4_chart(four_hr: list[tuple[int, int]], use_70pct: bool, slow_walkers: bool) -> go.Figure:
    """Scatter of pedestrian 4-hour volumes vs W4 curve."""
    if use_70pct:
        pts = W4_4HR_70PCT_SLOW if slow_walkers else W4_4HR_70PCT_STD
    else:
        pts = W4_4HR_URBAN_SLOW if slow_walkers else W4_4HR_URBAN_STD
    x_c = np.linspace(pts[0, 0], pts[-1, 0], 100)
    y_c = np.interp(x_c, pts[:, 0], pts[:, 1])
    colors = ["green" if w4_4hr_curve_check(maj, ped, use_70pct, slow_walkers) else "red"
              for maj, ped in four_hr]
    fig = go.Figure()
    fig.add_scatter(x=x_c, y=y_c, mode="lines", name="Warrant Curve", line_color="gray", line_dash="dash")
    fig.add_scatter(
        x=[maj for maj, _ in four_hr],
        y=[ped for _, ped in four_hr],
        mode="markers",
        marker=dict(size=12, color=colors),
        name="Study Hours",
    )
    fig.update_layout(
        title="Warrant 4 — Pedestrian 4-Hour Volume Plot (CA MUTCD 2026)",
        xaxis_title="Major Street VPH (Both Approaches)",
        yaxis_title="Pedestrians Crossing Major Street (PPH)",
        template="plotly_white",
    )
    return fig


# ==============================================================================
# SECTION 10: PDF REPORT GENERATION
# ==============================================================================

class WarrantPDF(FPDF):
    """FPDF2 subclass with consistent CA MUTCD header and footer."""

    _project_name: str = ""
    _cite: str = PDF_FOOTER_CITE

    def set_project(self, name: str) -> None:
        self._project_name = name

    def header(self) -> None:
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, "CA MUTCD 2026 — Traffic Signal Warrant Analysis", align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, self._project_name, align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_draw_color(150, 150, 150)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, self._cite, align="L")
        self.cell(0, 5, f"Page {self.page_no()} / {{nb}}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)


def _pdf_section_title(pdf: WarrantPDF, title: str) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(220, 230, 241)
    pdf.cell(0, 8, title, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def _pdf_kv(pdf: WarrantPDF, key: str, value: str) -> None:
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(55, 6, key + ":", new_x="RIGHT", new_y="LAST")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")


def _pdf_cover_page(pdf: WarrantPDF, proj: ProjectInfo, calc_date: str) -> None:
    pdf.add_page()
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Traffic Signal Warrant Analysis", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, proj.name, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    _pdf_section_title(pdf, "Project Information")
    _pdf_kv(pdf, "Location", proj.location)
    _pdf_kv(pdf, "County", proj.county)
    _pdf_kv(pdf, "Route/Road", proj.route)
    _pdf_kv(pdf, "District", proj.district)
    _pdf_kv(pdf, "Analyst", proj.analyst)
    _pdf_kv(pdf, "Checker", proj.checker)
    _pdf_kv(pdf, "Calc Date", proj.calc_date)
    _pdf_kv(pdf, "Check Date", proj.check_date)
    _pdf_kv(pdf, "Coordinates", f"Lat {proj.lat:.6f}, Lon {proj.lon:.6f}")
    pdf.ln(6)
    _pdf_section_title(pdf, "Reference Standard")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5,
        "All warrant thresholds, table values, and analysis criteria are from:\n"
        "California MUTCD 2026 Edition (FHWA MUTCD 2023, as amended for use in California)\n"
        "Chapter 4C — Traffic Control Signal Needs Studies  |  January 18, 2026\n"
        "Reference file: camutcd-2026-4c.pdf"
    )


def _pdf_warrant_page(pdf: WarrantPDF, result: WarrantResult) -> None:
    pdf.add_page()
    color = (0, 140, 0) if result.satisfied else (180, 0, 0)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*color)
    status_str = "SATISFIED ✓" if result.satisfied else "NOT SATISFIED ✗"
    pdf.cell(0, 9, f"Warrant {result.warrant_num} — {result.name}   [{status_str}]",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    for note in result.notes:
        pdf.multi_cell(0, 5, f"• {note}")
    pdf.ln(3)
    if result.detail_rows:
        _pdf_table(pdf, result.detail_rows)


def _pdf_table(pdf: WarrantPDF, rows: list[dict]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    col_w = min(35, (pdf.w - pdf.l_margin - pdf.r_margin) / len(headers))
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(200, 210, 230)
    for h in headers:
        pdf.cell(col_w, 6, str(h)[:16], border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for i, row in enumerate(rows):
        fill = (i % 2 == 0)
        if fill:
            pdf.set_fill_color(245, 247, 250)
        for h in headers:
            pdf.cell(col_w, 5, str(row.get(h, ""))[:16], border=1, fill=fill, align="C")
        pdf.ln()


def _pdf_chart_page(pdf: WarrantPDF, title: str, png_bytes: bytes) -> None:
    if not png_bytes:
        return
    pdf.add_page()
    _pdf_section_title(pdf, title)
    pdf.image(io.BytesIO(png_bytes), x=10, y=None, w=180)


def _pdf_accessibility_page(pdf: WarrantPDF, checked: dict[str, bool]) -> None:
    pdf.add_page()
    _pdf_section_title(pdf, "Pedestrian Accessibility Notes — CA MUTCD 2026 + PROWAG")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5,
        "The following checklist addresses CA MUTCD 2026 Chapter 6D, Chapter 4E, and PROWAG "
        "requirements for accessible pedestrian routes near traffic signal installations and "
        "construction zones."
    )
    pdf.ln(3)
    for item in PROWAG_CHECKLIST:
        status = checked.get(item, False)
        mark = "[✓]" if status else "[ ]"
        pdf.set_font("Helvetica", "B" if status else "", 9)
        pdf.multi_cell(0, 5, f"{mark}  {item}")
        pdf.ln(1)


def _pdf_summary_page(pdf: WarrantPDF, results: list[WarrantResult]) -> None:
    pdf.add_page()
    _pdf_section_title(pdf, "Summary of Warrant Analysis")
    satisfied = [r for r in results if r.satisfied]
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5,
        "The satisfaction of one or more traffic signal warrants does not in itself require "
        "the installation of a traffic control signal. The decision to install shall be based "
        "on an engineering study considering all factors. (CA MUTCD 2026 §4C.01)"
    )
    pdf.ln(4)
    col_w = (pdf.w - pdf.l_margin - pdf.r_margin) / 3
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(200, 210, 230)
    for h in ["Warrant", "Name", "Status"]:
        pdf.cell(col_w, 7, h, border=1, fill=True, align="C")
    pdf.ln()
    for r in results:
        color = (0, 140, 0) if r.satisfied else (180, 0, 0)
        pdf.set_text_color(*color)
        pdf.set_font("Helvetica", "B" if r.satisfied else "", 9)
        pdf.cell(col_w, 6, f"Warrant {r.warrant_num}", border=1, align="C")
        pdf.cell(col_w, 6, r.name[:30], border=1)
        pdf.cell(col_w, 6, "SATISFIED" if r.satisfied else "Not Satisfied", border=1, align="C")
        pdf.ln()
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 10)
    num_sat = len(satisfied)
    if num_sat == 0:
        rec = "No warrants are satisfied. Installation of a traffic control signal is not warranted at this time."
    elif num_sat <= 2:
        names = ", ".join(f"Warrant {r.warrant_num}" for r in satisfied)
        rec = (f"{names} {'is' if num_sat == 1 else 'are'} satisfied. "
               "Engineering judgment is required before proceeding with signal installation.")
    else:
        rec = (f"{num_sat} warrants are satisfied. Engineering study supports consideration "
               "of signal installation subject to further review.")
    pdf.multi_cell(0, 6, f"Recommendation: {rec}")


def build_pdf_report(
    proj: ProjectInfo,
    results: list[WarrantResult],
    accessibility_checked: dict[str, bool],
    chart_pngs: dict[str, bytes],
) -> bytes:
    """Assemble and return the full warrant analysis PDF as bytes."""
    pdf = WarrantPDF(orientation="P", unit="mm", format="Letter")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.alias_nb_pages()
    pdf.set_project(proj.name)
    _pdf_cover_page(pdf, proj, proj.calc_date)
    for r in results:
        _pdf_warrant_page(pdf, r)
    for title, png in chart_pngs.items():
        if png:
            _pdf_chart_page(pdf, title, png)
    _pdf_accessibility_page(pdf, accessibility_checked)
    _pdf_summary_page(pdf, results)
    return bytes(pdf.output())


# ==============================================================================
# SECTION 11: SESSION STATE INITIALISATION
# ==============================================================================

_DEFAULT_HOURS: list[HourlyVolumes] = [
    HourlyVolumes(h, 0, 0) for h in range(8)
]

def _init_session_state() -> None:
    """Initialise all session-state keys to defaults on first run."""
    defaults: dict = {
        "proj": ProjectInfo(),
        "tmc_data": None,
        "adt_data": None,
        "w1_hours": [HourlyVolumes(h) for h in range(8)],
        "w2_hours": [HourlyVolumes(h) for h in range(4)],
        "w3_peak": HourlyVolumes(7),
        "w1_result": None,
        "w2_result": None,
        "w3_result": None,
        "w4_result": None,
        "w5_result": None,
        "w6_result": None,
        "w7_result": None,
        "w8_result": None,
        "w9_result": None,
        "accessibility_checked": {item: False for item in PROWAG_CHECKLIST},
        "pdf_ref": {},
        "map_html": None,
        "w4_four_hr": [(0, 0)] * 4,
        "w4_peak": (0, 0),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ==============================================================================
# SECTION 12: SIDEBAR
# ==============================================================================

def render_sidebar(pdf_ref: dict) -> None:
    """Render sidebar with project info inputs and PDF status."""
    with st.sidebar:
        st.header("🚦 CA MUTCD 2026 Warrants")
        st.markdown("**Reference PDF Status**")
        status = pdf_ref.get("status", "not_found")
        if status == "loaded":
            pages = pdf_ref.get("total_pages", "?")
            st.success(f"✓ PDF loaded ({pages} pages)")
        elif status == "no_library":
            st.warning("⚠ PDF library not found")
        elif status == "error":
            st.error("✗ PDF parse error — using hardcoded thresholds")
        else:
            st.error("✗ PDF not found — using hardcoded CA MUTCD 2026 thresholds")

        st.markdown("---")
        st.subheader("Project Header")
        proj = st.session_state["proj"]
        proj.name       = st.text_input("Intersection Name",   value=proj.name,       key="in_proj_name")
        proj.location   = st.text_input("Location/Address",    value=proj.location,   key="in_proj_loc")
        proj.county     = st.text_input("County",              value=proj.county,     key="in_proj_county")
        proj.route      = st.text_input("Route/Road",          value=proj.route,      key="in_proj_route")
        proj.district   = st.text_input("District",            value=proj.district,   key="in_proj_dist")
        proj.analyst    = st.text_input("Analyst",             value=proj.analyst,    key="in_proj_analyst")
        proj.checker    = st.text_input("Checker",             value=proj.checker,    key="in_proj_checker")
        proj.calc_date  = st.text_input("Calc Date",           value=proj.calc_date,  key="in_proj_date")
        st.session_state["proj"] = proj


# ==============================================================================
# SECTION 13: DATA IMPORT TAB
# ==============================================================================

def render_data_import_tab() -> None:
    st.header("📁 Data Import")
    col_adt, col_tmc = st.columns(2)

    with col_adt:
        st.subheader("ADT CSV Import")
        st.caption("Format: Timestamp, Direction, Volume  (15-min intervals)")
        k_factor = st.number_input("K-Factor (ADT → peak-hour ratio)", value=DEFAULT_K_FACTOR,
                                   min_value=0.01, max_value=0.30, step=0.01, key="in_k_factor")
        adt_file = st.file_uploader("Upload ADT CSV", type=["csv"], key="adt_uploader")
        if adt_file is not None:
            try:
                st.session_state["adt_data"] = parse_adt_csv(adt_file, k_factor)
                st.success("ADT CSV loaded successfully.")
            except Exception as e:
                st.error(f"ADT parse error: {e}")
        if st.session_state.get("adt_data"):
            adt: ADTData = st.session_state["adt_data"]
            st.dataframe(adt.hourly_df, height=220)
            total_adt = adt.raw_df["Volume"].sum() if "Volume" in adt.raw_df.columns else 0
            st.metric("Total ADT", f"{total_adt:,} vpd")
            st.metric("Peak Hour Estimate", f"{round(total_adt * k_factor):,} vph (K={k_factor})")

    with col_tmc:
        st.subheader("TMC CSV Import (15-min counts)")
        st.caption("Format: Timestamp, Approach, Movement, Volume, Ped, Bike")
        tmc_file = st.file_uploader("Upload TMC CSV", type=["csv"], key="tmc_uploader")
        if tmc_file is not None:
            try:
                st.session_state["tmc_data"] = parse_tmc_csv(tmc_file)
                st.success("TMC CSV loaded. Peak hours auto-detected.")
            except Exception as e:
                st.error(f"TMC parse error: {e}")
        if st.session_state.get("tmc_data"):
            tmc: TMCData = st.session_state["tmc_data"]
            st.dataframe(tmc.hourly_df, height=220)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Peak Hour", f"{tmc.peak_1hr:02d}:00")
            col2.metric("PHF", f"{tmc.phf:.3f}")
            col3.metric("Peak 4 Hrs", str(tmc.peak_4hrs))
            col4.metric("Peak 8 Hrs", str(tmc.peak_8hrs[:4]) + "…")

    st.divider()
    st.subheader("Auto-populate Warrant Volumes from TMC Data")
    if st.session_state.get("tmc_data") and st.button("↩ Push TMC Data to All Warrants"):
        _push_tmc_to_warrants(st.session_state["tmc_data"])
        st.success("Warrant volume inputs updated from TMC data.")


def _push_tmc_to_warrants(tmc: TMCData) -> None:
    """Populate warrant volume session-state from TMC auto-detected peaks."""
    hourly = tmc.hourly_df.set_index("Hour")

    def get_vol(h: int) -> HourlyVolumes:
        row = hourly.loc[h] if h in hourly.index else None
        return HourlyVolumes(
            hour=h,
            major_vph=int(row["TotalVehicles"] * 0.7) if row is not None else 0,
            minor_vph=int(row["TotalVehicles"] * 0.3) if row is not None else 0,
            ped_vph=int(row["TotalPed"]) if row is not None else 0,
            bike_vph=int(row["TotalBike"]) if row is not None else 0,
        )

    st.session_state["w1_hours"] = [get_vol(h) for h in tmc.peak_8hrs]
    st.session_state["w2_hours"] = [get_vol(h) for h in tmc.peak_4hrs]
    peak = tmc.peak_1hr
    st.session_state["w3_peak"] = get_vol(peak)
    st.session_state["w4_four_hr"] = [
        (get_vol(h).major_vph, get_vol(h).ped_vph) for h in tmc.peak_4hrs
    ]
    ph_vol = get_vol(peak)
    st.session_state["w4_peak"] = (ph_vol.major_vph, ph_vol.ped_vph)


# ==============================================================================
# SECTION 14: MAP TAB
# ==============================================================================

def render_map_tab() -> None:
    st.header("🗺 Intersection Map")
    col_inp, col_map = st.columns([1, 2])

    with col_inp:
        proj = st.session_state["proj"]
        lat = st.number_input("Latitude", value=proj.lat, format="%.6f", key="in_map_lat")
        lon = st.number_input("Longitude", value=proj.lon, format="%.6f", key="in_map_lon")
        name = st.text_input("Intersection Label", value=proj.name, key="in_map_name")
        st.session_state["proj"].lat = lat
        st.session_state["proj"].lon = lon

    with col_map:
        tmc: TMCData | None = st.session_state.get("tmc_data")
        hourly_df = tmc.hourly_df if tmc else None
        m = render_folium_map(lat, lon, name, hourly_df)
        result = st_folium(m, width=600, height=450, returned_objects=[])
        st.caption(
            "💡 Map uses OpenStreetMap & Esri World Imagery layers. "
            "Toggle layers via the layer control (top-right). "
            "The folium HTML can be saved using your browser's Save function for the PDF if needed."
        )

    # Save map HTML for reference
    m_html = m.get_root().render()
    st.session_state["map_html"] = m_html
    if st.download_button("⬇ Download Map HTML", m_html.encode(), file_name="intersection_map.html", mime="text/html"):
        pass  # download triggered


# ==============================================================================
# SECTION 15: SHARED LANE CONFIG WIDGET
# ==============================================================================

def _lane_config_inputs(prefix: str) -> tuple[LaneConfig, float, bool]:
    """Render lane config, speed, and population widgets. Return (LaneConfig, speed, pop_lt10k)."""
    c1, c2, c3, c4 = st.columns(4)
    major_lanes = c1.number_input("Major St Lanes", 1, 6, 2, key=f"{prefix}_maj_ln")
    minor_lanes = c2.number_input("Minor St Lanes", 1, 4, 1, key=f"{prefix}_min_ln")
    speed = c3.number_input("Major St Speed (mph)", 20, 75, 35, key=f"{prefix}_spd")
    pop_lt10k = c4.checkbox("Pop < 10,000 (rural/isolated)", key=f"{prefix}_pop")
    return LaneConfig(int(major_lanes), int(minor_lanes)), float(speed), pop_lt10k


# ==============================================================================
# SECTION 16: WARRANT 1 TAB
# ==============================================================================

def render_w1_tab() -> None:
    st.header("Warrant 1 — Eight-Hour Vehicular Volume")
    st.caption("CA MUTCD 2026 Section 4C.02, Table 4C-1 (p.1074)")

    lane_cfg, speed, pop_lt10k = _lane_config_inputs("w1")
    use_70 = _use_70pct(speed, pop_lt10k)
    col_factor = "70%" if use_70 else "100%"
    st.info(f"Using **{col_factor}** threshold columns (speed={speed} mph, pop<10K={pop_lt10k})")

    combo_trial = st.checkbox("Adequate trial of alternatives completed (needed for Combination A+B)", key="w1_combo_trial")

    st.subheader("Enter 8 Representative Hourly Volumes")
    cols = st.columns(8)
    hours: list[HourlyVolumes] = []
    defaults = st.session_state.get("w1_hours", [HourlyVolumes(h) for h in range(8)])
    for i, col in enumerate(cols):
        d = defaults[i] if i < len(defaults) else HourlyVolumes(i)
        with col:
            st.markdown(f"**Hr {i+1}**")
            hr = st.number_input("Hour", 0, 23, d.hour, key=f"w1_hr_{i}")
            maj = st.number_input("Major", 0, 9999, d.major_vph, key=f"w1_maj_{i}")
            mn = st.number_input("Minor", 0, 9999, d.minor_vph, key=f"w1_min_{i}")
        hours.append(HourlyVolumes(int(hr), int(maj), int(mn)))

    st.session_state["w1_hours"] = hours

    if st.button("▶ Evaluate Warrant 1", key="btn_w1"):
        result = eval_w1(hours, lane_cfg, speed, pop_lt10k, combo_trial)
        st.session_state["w1_result"] = result
        _show_warrant_result(result)
        chart = make_w1_chart(hours, lane_cfg, use_70)
        st.plotly_chart(chart, use_container_width=True)
    elif st.session_state.get("w1_result"):
        _show_warrant_result(st.session_state["w1_result"])

    _show_w1_reference_table(lane_cfg, use_70)


def _show_w1_reference_table(lane_cfg: LaneConfig, use_70: bool) -> None:
    """Display threshold reference table for current lane configuration."""
    st.subheader("Reference Thresholds — Table 4C-1 (CA MUTCD 2026)")
    row_a = W1_COND_A[lane_cfg.key]
    row_b = W1_COND_B[lane_cfg.key]
    col_labels = ["100%", "80%", "70%", "56%"]
    highlight = 2 if use_70 else 0
    df_ref = pd.DataFrame({
        "Column": col_labels,
        "Cond A — Major VPH": row_a[:4],
        "Cond A — Minor VPH": row_a[4:],
        "Cond B — Major VPH": row_b[:4],
        "Cond B — Minor VPH": row_b[4:],
        "Applied": ["→ ACTIVE ←" if i == highlight else "" for i in range(4)],
    })
    st.dataframe(df_ref, hide_index=True)


# ==============================================================================
# SECTION 17: WARRANT 2 TAB
# ==============================================================================

def render_w2_tab() -> None:
    st.header("Warrant 2 — Four-Hour Vehicular Volume")
    st.caption("CA MUTCD 2026 Section 4C.03, Figures 4C-1 (urban) / 4C-2 (70% factor)")

    lane_cfg, speed, pop_lt10k = _lane_config_inputs("w2")
    use_70 = _use_70pct(speed, pop_lt10k)
    fig_ref = "Figure 4C-2 (70% factor)" if use_70 else "Figure 4C-1 (urban)"
    st.info(f"Checking against **{fig_ref}**")

    st.subheader("Enter 4 Highest Hourly Volumes")
    cols = st.columns(4)
    four_hrs: list[HourlyVolumes] = []
    defaults = st.session_state.get("w2_hours", [HourlyVolumes(h) for h in range(4)])
    for i, col in enumerate(cols):
        d = defaults[i] if i < len(defaults) else HourlyVolumes(i)
        with col:
            st.markdown(f"**Hour {i+1}**")
            hr = st.number_input("Hour", 0, 23, d.hour, key=f"w2_hr_{i}")
            maj = st.number_input("Major VPH", 0, 9999, d.major_vph, key=f"w2_maj_{i}")
            mn = st.number_input("Minor VPH", 0, 9999, d.minor_vph, key=f"w2_min_{i}")
        four_hrs.append(HourlyVolumes(int(hr), int(maj), int(mn)))

    st.session_state["w2_hours"] = four_hrs

    if st.button("▶ Evaluate Warrant 2", key="btn_w2"):
        result = eval_w2(four_hrs, lane_cfg, use_70)
        st.session_state["w2_result"] = result
        _show_warrant_result(result)
        chart = make_w2_chart(four_hrs, lane_cfg, use_70)
        st.plotly_chart(chart, use_container_width=True)
    elif st.session_state.get("w2_result"):
        _show_warrant_result(st.session_state["w2_result"])


# ==============================================================================
# SECTION 18: WARRANT 3 TAB
# ==============================================================================

def render_w3_tab() -> None:
    st.header("Warrant 3 — Peak Hour")
    st.caption("CA MUTCD 2026 Section 4C.04, Figures 4C-3 / 4C-4")

    lane_cfg, speed, pop_lt10k = _lane_config_inputs("w3")
    use_70 = _use_70pct(speed, pop_lt10k)

    st.subheader("Part A — Delay + Volume + Entering Volume")
    c1, c2, c3 = st.columns(3)
    delay = c1.number_input("Minor-street delay (veh-hrs/hr)", 0.0, 50.0, 0.0, 0.1, key="w3_delay")
    entering = c2.number_input("Total entering volume (vph)", 0, 5000, 0, key="w3_entering")
    num_app = c3.radio("Number of approaches", [3, 4], horizontal=True, key="w3_approaches")

    st.subheader("Part B — Peak Hour Volume (Graphical Curve)")
    d = st.session_state.get("w3_peak", HourlyVolumes(7))
    c1, c2, c3 = st.columns(3)
    hr = c1.number_input("Peak Hour", 0, 23, d.hour, key="w3_hr")
    maj = c2.number_input("Major Street VPH (both approaches)", 0, 9999, d.major_vph, key="w3_maj")
    mn = c3.number_input("Minor Street VPH (higher approach)", 0, 9999, d.minor_vph, key="w3_min")
    peak_vol = HourlyVolumes(int(hr), int(maj), int(mn))
    st.session_state["w3_peak"] = peak_vol

    if st.button("▶ Evaluate Warrant 3", key="btn_w3"):
        result = eval_w3(peak_vol, lane_cfg, use_70, delay, int(entering), int(num_app))
        st.session_state["w3_result"] = result
        _show_warrant_result(result)
        # Peak hour curve chart
        curves = W3_CURVES_70PCT if use_70 else W3_CURVES_URBAN
        key = lane_cfg.key if lane_cfg.key in curves else (1, 1)
        pts = curves[key]
        x_c = np.linspace(pts[0, 0], pts[-1, 0], 100)
        y_c = np.interp(x_c, pts[:, 0], pts[:, 1])
        fig = go.Figure()
        fig.add_scatter(x=x_c, y=y_c, mode="lines", name="Warrant Curve", line_color="gray", line_dash="dash")
        above = w3_curve_check(maj, mn, lane_cfg, use_70)
        fig.add_scatter(x=[maj], y=[mn], mode="markers",
                        marker=dict(size=14, color="green" if above else "red"),
                        name=f"{'Above' if above else 'Below'} curve")
        fig.update_layout(
            title="Warrant 3 — Peak Hour Curve", xaxis_title="Major VPH", yaxis_title="Minor VPH",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    elif st.session_state.get("w3_result"):
        _show_warrant_result(st.session_state["w3_result"])


# ==============================================================================
# SECTION 19: WARRANT 4 TAB
# ==============================================================================

def render_w4_tab() -> None:
    st.header("Warrant 4 — Pedestrian Volume")
    st.caption("CA MUTCD 2026 Section 4C.05, Figures 4C-5 through 4C-8")

    c1, c2, c3 = st.columns(3)
    speed = c1.number_input("Major St Speed (mph)", 20, 75, 35, key="w4_spd")
    pop_lt10k = c2.checkbox("Pop < 10,000 or > 40 mph", key="w4_pop")
    slow_walk = c3.checkbox("Slow walkers (15th-pct < 3.5 fps)", key="w4_slow")
    use_70 = _use_70pct(speed, pop_lt10k)

    dist_ft = st.number_input("Distance to nearest signal (ft)", 0, 5000, 500, key="w4_dist")
    no_restrict = st.checkbox("Signal will NOT restrict progressive traffic flow", key="w4_norestrict")

    st.subheader("Part 1A — Pedestrian 4-Hour Volumes")
    four_hr_pairs: list[tuple[int, int]] = []
    cols = st.columns(4)
    defaults_4hr = st.session_state.get("w4_four_hr", [(0, 0)] * 4)
    for i, col in enumerate(cols):
        d_maj, d_ped = defaults_4hr[i] if i < len(defaults_4hr) else (0, 0)
        with col:
            st.markdown(f"**Hour {i+1}**")
            maj = st.number_input("Major VPH", 0, 9999, d_maj, key=f"w4_4hr_maj_{i}")
            ped = st.number_input("Ped VPH", 0, 9999, d_ped, key=f"w4_4hr_ped_{i}")
            four_hr_pairs.append((int(maj), int(ped)))

    st.subheader("Part 1B — Pedestrian Peak-Hour Volume")
    d_pk = st.session_state.get("w4_peak", (0, 0))
    cp1, cp2 = st.columns(2)
    pk_maj = cp1.number_input("Major VPH (peak hr)", 0, 9999, d_pk[0], key="w4_pk_maj")
    pk_ped = cp2.number_input("Ped VPH (peak hr)", 0, 9999, d_pk[1], key="w4_pk_ped")
    st.session_state["w4_four_hr"] = four_hr_pairs
    st.session_state["w4_peak"] = (int(pk_maj), int(pk_ped))

    if st.button("▶ Evaluate Warrant 4", key="btn_w4"):
        result = eval_w4(
            four_hr_pairs, (int(pk_maj), int(pk_ped)),
            use_70, slow_walk, float(dist_ft), no_restrict,
        )
        st.session_state["w4_result"] = result
        _show_warrant_result(result)
        chart = make_w4_chart(four_hr_pairs, use_70, slow_walk)
        st.plotly_chart(chart, use_container_width=True)
    elif st.session_state.get("w4_result"):
        _show_warrant_result(st.session_state["w4_result"])


# ==============================================================================
# SECTION 20: WARRANT 5 TAB
# ==============================================================================

def render_w5_tab() -> None:
    st.header("Warrant 5 — School Crossing")
    st.caption("CA MUTCD 2026 Section 4C.06")

    c1, c2 = st.columns(2)
    gaps = c1.number_input("Adequate gaps per hour", 0, 200, 10, key="w5_gaps")
    children = c2.number_input("School children crossing per hour", 0, 500, 15, key="w5_children")
    alternatives = st.checkbox("Less-restrictive alternatives have been tried", key="w5_alts")

    st.subheader("Part B — Signal Spacing")
    dist_ft = st.number_input("Distance to nearest signal (ft)", 0, 5000, 400, key="w5_dist")
    no_restrict = st.checkbox("Signal will NOT restrict progressive traffic flow", key="w5_norestrict")

    if st.button("▶ Evaluate Warrant 5", key="btn_w5"):
        result = eval_w5(int(gaps), int(children), alternatives, float(dist_ft), no_restrict)
        st.session_state["w5_result"] = result
        _show_warrant_result(result)
    elif st.session_state.get("w5_result"):
        _show_warrant_result(st.session_state["w5_result"])

    with st.expander("ℹ Section 4C.06 Requirements Summary"):
        st.markdown("""
        **Part A** — All three must be met for the same 1-hour period:
        1. Number of adequate gaps < number of children crossing
        2. Children using crossing > 20 per hour
        3. Less-restrictive alternatives have been considered and tried

        **Part B** — Either condition:
        - Distance to nearest signal along major street > 300 ft, OR
        - Proposed signal will not restrict progressive traffic flow
        """)


# ==============================================================================
# SECTION 21: WARRANT 6 TAB
# ==============================================================================

def render_w6_tab() -> None:
    st.header("Warrant 6 — Coordinated Signal System")
    st.caption("CA MUTCD 2026 Section 4C.07")

    st.subheader("Distance to Nearest Signals")
    c1, c2, c3, c4 = st.columns(4)
    dist_n = c1.number_input("North (ft)", 0, 99999, 1200, key="w6_dn")
    dist_s = c2.number_input("South (ft)", 0, 99999, 1200, key="w6_ds")
    dist_e = c3.number_input("East (ft)",  0, 99999, 1200, key="w6_de")
    dist_w = c4.number_input("West (ft)",  0, 99999, 1200, key="w6_dw")
    min_dist = min(dist_n, dist_s, dist_e, dist_w)
    st.metric("Minimum Distance", f"{min_dist} ft", delta=f"{'> 1000 ft ✓' if min_dist > 1000 else '≤ 1000 ft ✗'}")

    one_way = st.checkbox("One-way street: adjacent signals too far apart for vehicular platooning", key="w6_oneway")
    two_way = st.checkbox("Two-way street: proposed + adjacent signals will provide progressive operation", key="w6_twoway")

    if st.button("▶ Evaluate Warrant 6", key="btn_w6"):
        result = eval_w6(float(min_dist), one_way, two_way)
        st.session_state["w6_result"] = result
        _show_warrant_result(result)
    elif st.session_state.get("w6_result"):
        _show_warrant_result(st.session_state["w6_result"])


# ==============================================================================
# SECTION 22: WARRANT 7 TAB
# ==============================================================================

def render_w7_tab() -> None:
    st.header("Warrant 7 — Crash Experience")
    st.caption("CA MUTCD 2026 Section 4C.08, Tables 4C-2 through 4C-5")

    lane_cfg, speed, pop_lt10k = _lane_config_inputs("w7")
    use_70 = _use_70pct(speed, pop_lt10k)
    alternatives = st.checkbox("Adequate trial of remedial alternatives has failed", key="w7_alts")

    st.subheader("Crash History")
    c1, c2 = st.columns(2)
    correctable = c1.number_input("Correctable crashes (injury + reportable damage) in past 12 months",
                                  0, 999, 0, key="w7_crashes")
    num_legs = c2.radio("Intersection type", [4, 3], horizontal=True, key="w7_legs",
                        captions=["4-leg (typical)", "3-leg (T-intersection)"])

    st.subheader("Crash Table Condition")
    st.caption("Select applicable crash table condition (Tables 4C-2 through 4C-5)")

    table_key = lane_cfg.key
    if use_70:
        t3yr = W7_CRASH_3YR_RURAL[table_key]
        t1yr = W7_CRASH_1YR_RURAL[table_key]
        label_3yr = "Table 4C-5 (3-yr rural/70%)"
        label_1yr = "Table 4C-4 (1-yr rural/70%)"
    else:
        t3yr = W7_CRASH_3YR_URBAN[table_key]
        t1yr = (6, 5, 4, 4)  # Table 4C-2 urban 1-yr (all combos same)
        label_3yr = "Table 4C-3 (3-yr urban)"
        label_1yr = "Table 4C-2 (1-yr urban)"

    leg_i = 0 if num_legs == 4 else 1
    fi_i = 2 if num_legs == 4 else 3

    df_thr = pd.DataFrame({
        "Table": [label_1yr, label_1yr, label_3yr, label_3yr],
        "Type": ["Total angle+ped (all sev)", "Fatal-and-injury angle+ped",
                 "Total angle+ped (all sev)", "Fatal-and-injury angle+ped"],
        "Threshold": [t1yr[leg_i], t1yr[fi_i], t3yr[leg_i], t3yr[fi_i]],
    })
    st.dataframe(df_thr, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    crashes_1yr_total = c1.number_input(f"{label_1yr} — Total crashes (1yr)", 0, 999, 0, key="w7_1yr_tot")
    crashes_1yr_fi = c2.number_input(f"{label_1yr} — F&I crashes (1yr)", 0, 999, 0, key="w7_1yr_fi")
    crashes_3yr_total = c3.number_input(f"{label_3yr} — Total crashes (3yr)", 0, 999, 0, key="w7_3yr_tot")
    crashes_3yr_fi = c4.number_input(f"{label_3yr} — F&I crashes (3yr)", 0, 999, 0, key="w7_3yr_fi")

    crash_table_ok = (
        crashes_1yr_total >= t1yr[leg_i] or
        crashes_1yr_fi >= t1yr[fi_i] or
        crashes_3yr_total >= t3yr[leg_i] or
        crashes_3yr_fi >= t3yr[fi_i]
    )

    st.subheader("W1 80% Volume Condition")
    w1_80_met = st.checkbox(
        "For each of any 8 hours: 80% of W1 Condition A OR 80% of Condition B volumes met",
        key="w7_w1_80pct",
    )
    st.caption("80% columns per Table 4C-1 (CA MUTCD 2026 p.1074): "
               f"Cond A major≥{W1_COND_A[lane_cfg.key][1]} minor≥{W1_COND_A[lane_cfg.key][5]}; "
               f"Cond B major≥{W1_COND_B[lane_cfg.key][1]} minor≥{W1_COND_B[lane_cfg.key][5]}")

    if st.button("▶ Evaluate Warrant 7", key="btn_w7"):
        result = eval_w7(int(correctable), crash_table_ok, alternatives, w1_80_met)
        st.session_state["w7_result"] = result
        _show_warrant_result(result)
    elif st.session_state.get("w7_result"):
        _show_warrant_result(st.session_state["w7_result"])


# ==============================================================================
# SECTION 23: WARRANT 8 TAB
# ==============================================================================

def render_w8_tab() -> None:
    st.header("Warrant 8 — Roadway Network")
    st.caption("CA MUTCD 2026 Section 4C.09")

    st.subheader("Functional Classification")
    principal = st.checkbox("Major street is a principal through route (connects areas of principal traffic generation)", key="w8_principal")
    rural_sub = st.checkbox("Major street is a rural or suburban highway (outside, entering, or traversing a city)", key="w8_rural")
    official_plan = st.checkbox("Major street appears as a major route on an official plan (e.g., urban transportation study)", key="w8_plan")

    st.subheader("Minimum Volume Conditions")
    lane_cfg, speed, pop_lt10k = _lane_config_inputs("w8")
    use_70 = _use_70pct(speed, pop_lt10k)
    maj_idx = 0  # Use 100% column for W8
    thr_a = W1_COND_A[lane_cfg.key]
    st.caption(
        f"Minimum volumes (W1 Cond A thresholds): "
        f"Major ≥ {thr_a[maj_idx]} vph, Minor ≥ {thr_a[maj_idx + 4]} vph  (Table 4C-1)"
    )
    c1, c2 = st.columns(2)
    major_vph = c1.number_input("Major street VPH (typical peak hour)", 0, 9999, 0, key="w8_maj")
    minor_vph = c2.number_input("Minor street VPH (higher approach)", 0, 9999, 0, key="w8_min")
    min_vols_ok = major_vph >= thr_a[maj_idx] and minor_vph >= thr_a[maj_idx + 4]

    if st.button("▶ Evaluate Warrant 8", key="btn_w8"):
        result = eval_w8(principal, rural_sub, official_plan, min_vols_ok)
        st.session_state["w8_result"] = result
        _show_warrant_result(result)
    elif st.session_state.get("w8_result"):
        _show_warrant_result(st.session_state["w8_result"])


# ==============================================================================
# SECTION 24: WARRANT 9 TAB
# ==============================================================================

def render_w9_tab() -> None:
    st.header("Warrant 9 — Intersection Near a Grade Crossing")
    st.caption("CA MUTCD 2026 Section 4C.10, Tables 4C-6/7/8, Figures 4C-9/4C-10")
    st.info(
        "This warrant applies when none of the other eight warrants are met but proximity "
        "to a grade crossing on a STOP/YIELD controlled approach is the principal concern."
    )

    st.subheader("Part A — Grade Crossing Proximity")
    c1, c2 = st.columns(2)
    track_dist = c1.number_input("Distance from stop/yield line to nearest track center (ft)", 0, 500, 100, key="w9_dist")
    two_lanes = c2.checkbox("2 or more approach lanes at the track crossing", key="w9_2lanes")
    fig_used = "Figure 4C-10" if two_lanes else "Figure 4C-9"
    st.caption(f"Applicable figure: {fig_used} (CA MUTCD 2026, p.1065–1066)")

    st.subheader("Part B — Volume During Peak Rail-Traffic Hour")
    c1, c2 = st.columns(2)
    major_vph = c1.number_input("Major street VPH (both approaches)", 0, 9999, 400, key="w9_maj")
    minor_vph = c2.number_input("Minor street approach VPH (crossing the track, 1 direction)", 0, 9999, 50, key="w9_min")

    st.subheader("Adjustment Factors")
    col1, col2, col3 = st.columns(3)
    rail_key = col1.selectbox("Daily rail traffic frequency (Table 4C-6)", list(W9_RAIL_AF.keys()), index=2, key="w9_rail")
    bus_key = col2.selectbox("% High-occupancy buses (Table 4C-7)", list(W9_BUS_AF.keys()), index=0, key="w9_bus")
    truck_key = col3.selectbox("% Tractor-trailer trucks, D≥70 ft (Table 4C-8)", list(W9_TRUCK_AF_D70.keys()), index=2, key="w9_truck")

    af = _compute_w9_af(rail_key, bus_key, truck_key)
    adjusted = minor_vph * af
    st.metric("Combined Adjustment Factor", f"{af:.3f}")
    st.metric("Adjusted Minor Approach VPH", f"{adjusted:.1f}")

    if st.button("▶ Evaluate Warrant 9", key="btn_w9"):
        result = eval_w9(float(track_dist), int(major_vph), int(minor_vph), rail_key, bus_key, truck_key, two_lanes)
        st.session_state["w9_result"] = result
        _show_warrant_result(result)
    elif st.session_state.get("w9_result"):
        _show_warrant_result(st.session_state["w9_result"])

    with st.expander("ℹ Adjustment Factor Tables (CA MUTCD 2026)"):
        c1, c2, c3 = st.columns(3)
        c1.subheader("Table 4C-6: Rail Frequency")
        c1.dataframe(pd.DataFrame(list(W9_RAIL_AF.items()), columns=["Trains/Day", "AF"]), hide_index=True)
        c2.subheader("Table 4C-7: Bus %")
        c2.dataframe(pd.DataFrame(list(W9_BUS_AF.items()), columns=["Bus %", "AF"]), hide_index=True)
        c3.subheader("Table 4C-8: Truck % (D≥70 ft)")
        c3.dataframe(pd.DataFrame(list(W9_TRUCK_AF_D70.items()), columns=["Truck %", "AF"]), hide_index=True)


# ==============================================================================
# SECTION 25: ACCESSIBILITY TAB
# ==============================================================================

def render_accessibility_tab() -> None:
    st.header("♿ Pedestrian Accessibility Notes")
    st.caption("CA MUTCD 2026 + PROWAG Compliance Checklist")

    st.markdown("""
    The following checklist addresses requirements from:
    - **CA MUTCD 2026** Chapter 4E (Pedestrian Signals), Chapter 6D (TTC Zones)
    - **PROWAG** (Public Rights-of-Way Accessibility Guidelines) §R304, §R205
    - **ADA Standards** for temporary pedestrian facilities
    """)

    checked = st.session_state.get("accessibility_checked", {item: False for item in PROWAG_CHECKLIST})
    for item in PROWAG_CHECKLIST:
        checked[item] = st.checkbox(item, value=checked.get(item, False), key=f"acc_{hash(item)}")
    st.session_state["accessibility_checked"] = checked

    n_checked = sum(checked.values())
    n_total = len(PROWAG_CHECKLIST)
    st.progress(n_checked / n_total, text=f"{n_checked}/{n_total} items confirmed")

    if n_checked == n_total:
        st.success("✓ All accessibility requirements confirmed.")
    elif n_checked >= n_total * 0.7:
        st.warning(f"⚠ {n_total - n_checked} items remain unchecked.")
    else:
        st.error(f"✗ {n_total - n_checked} accessibility items need attention.")

    with st.expander("📋 CA MUTCD 2026 Accessibility Reference"):
        st.markdown("""
        **Continuous Accessible Route (CA MUTCD 2026 §6D.01 & PROWAG §R301)**
        - Pedestrian access routes (PAR) must be provided through or around work zones
        - Minimum clear width: 4.0 ft (48 in); 5.0 ft preferred
        - Maximum cross-slope: 2% (1:50); maximum running slope = adjacent sidewalk grade

        **Curb Ramps & Blended Transitions (PROWAG §R304)**
        - Required at every location where pedestrian route crosses a curb
        - Detectable warning surface (truncated domes, 60-in depth min)
        - Landing at top: min 36 in × 60 in clear space

        **Accessible Pedestrian Signals (CA MUTCD 2026 §4E.06)**
        - When warranted, APS must provide audible walk indication and vibrotactile arrow
        - Must be maintained or replaced during construction; temporary APS if relocated

        **Pedestrian Detour Signing (CA MUTCD 2026 Chapter 6D)**
        - R9-3A "SIDEWALK CLOSED — USE OTHER SIDE" or R9-11 "PEDESTRIAN DETOUR" signs
        - Directional arrows at all decision points along detour route
        - Signs must be at wheelchair-accessible height (bottom 7 ft max on fixed supports)

        **PROWAG §R205 — Alterations**
        - When public rights-of-way are altered, all altered elements must comply with PROWAG
        - If full compliance creates extraordinary hardship, maximum feasible compliance required
        """)


# ==============================================================================
# SECTION 26: SHARED RESULT DISPLAY
# ==============================================================================

def _show_warrant_result(result: WarrantResult) -> None:
    """Display a pass/fail badge and notes for a WarrantResult."""
    if result.satisfied:
        st.success(f"✅ Warrant {result.warrant_num} — {result.name}: **SATISFIED**")
    else:
        st.error(f"❌ Warrant {result.warrant_num} — {result.name}: **NOT SATISFIED**")
    with st.expander("Analysis Details", expanded=result.satisfied):
        for note in result.notes:
            st.markdown(f"- {note}")
        if result.detail_rows:
            st.dataframe(pd.DataFrame(result.detail_rows), hide_index=True)


# ==============================================================================
# SECTION 27: SUMMARY TAB
# ==============================================================================

def render_summary_tab() -> None:
    st.header("📊 Summary of Warrant Analysis")
    st.caption("CA MUTCD 2026 Chapter 4C — The satisfaction of a warrant does NOT require signal installation.")

    result_keys = [f"w{i}_result" for i in range(1, 10)]
    results: list[WarrantResult] = [
        st.session_state[k] for k in result_keys if st.session_state.get(k)
    ]

    if not results:
        st.info("No warrants evaluated yet. Complete each Warrant tab and click Evaluate.")
        return

    rows = []
    for r in results:
        rows.append({
            "Warrant": f"W{r.warrant_num}",
            "Name": r.name,
            "Status": "SATISFIED ✓" if r.satisfied else "Not Satisfied ✗",
            "Notes": r.notes[0] if r.notes else "",
        })
    df_summary = pd.DataFrame(rows)

    def style_status(val: str) -> str:
        if "SATISFIED" in val and "Not" not in val:
            return "color: green; font-weight: bold"
        return "color: red"

    st.dataframe(
        df_summary.style.map(style_status, subset=["Status"]),
        hide_index=True, use_container_width=True,
    )

    satisfied_count = sum(1 for r in results if r.satisfied)
    col1, col2, col3 = st.columns(3)
    col1.metric("Warrants Evaluated", len(results))
    col2.metric("Warrants Satisfied", satisfied_count)
    col3.metric("Warrants Not Satisfied", len(results) - satisfied_count)

    if satisfied_count == 0:
        st.warning("No warrants are currently satisfied.")
    else:
        names = [f"W{r.warrant_num}" for r in results if r.satisfied]
        st.success(f"Satisfied: {', '.join(names)}")

    st.divider()
    _render_summary_exports(results, df_summary)


def _render_summary_exports(results: list[WarrantResult], df_summary: pd.DataFrame) -> None:
    st.subheader("Export")
    col_csv, col_pdf = st.columns(2)

    with col_csv:
        csv_bytes = df_summary.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download Summary CSV",
            csv_bytes,
            file_name="warrant_summary.csv",
            mime="text/csv",
            key="dl_csv",
        )

    with col_pdf:
        if st.button("🖨 Generate PDF Report", key="btn_pdf"):
            _generate_and_download_pdf(results)


def _generate_and_download_pdf(results: list[WarrantResult]) -> None:
    """Build PDF report and offer download button."""
    proj = st.session_state.get("proj", ProjectInfo())
    acc_checked = st.session_state.get("accessibility_checked", {})

    chart_pngs: dict[str, bytes] = {}
    if st.session_state.get("w1_result") and st.session_state.get("w1_hours"):
        lane_cfg = LaneConfig(2, 1)
        use_70 = False
        fig1 = make_w1_chart(st.session_state["w1_hours"], lane_cfg, use_70)
        chart_pngs["Warrant 1 — 8-Hour Volume Chart"] = _fig_to_png(fig1)

    if st.session_state.get("w2_result") and st.session_state.get("w2_hours"):
        lane_cfg = LaneConfig(2, 1)
        fig2 = make_w2_chart(st.session_state["w2_hours"], lane_cfg, False)
        chart_pngs["Warrant 2 — Four-Hour Volume Chart"] = _fig_to_png(fig2)

    with st.spinner("Generating PDF report…"):
        try:
            pdf_bytes = build_pdf_report(proj, results, acc_checked, chart_pngs)
            st.download_button(
                "⬇ Download PDF Report",
                pdf_bytes,
                file_name=f"warrant_analysis_{proj.calc_date}.pdf",
                mime="application/pdf",
                key="dl_pdf",
            )
            st.success("PDF report ready.")
        except Exception as exc:
            st.error(f"PDF generation error: {exc}")
            logger.error("PDF error", exc_info=True)


# ==============================================================================
# SECTION 28: MAIN
# ==============================================================================

def main() -> None:
    st.set_page_config(
        page_title="CA MUTCD 2026 — Traffic Signal Warrant Analysis",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session_state()

    # Load PDF reference once per session
    if not st.session_state.get("pdf_ref"):
        st.session_state["pdf_ref"] = load_camutcd_reference(PDF_REFERENCE_PATH)

    render_sidebar(st.session_state["pdf_ref"])

    st.title("🚦 CA MUTCD 2026 — Traffic Signal Warrant Analysis")
    st.caption(
        "Replicates the Caltrans Traffic Signal Warrants Worksheet (Figure 4C-101(CA)) "
        "with thresholds from CA MUTCD 2026, Chapter 4C (January 18, 2026)."
    )

    tabs = st.tabs([
        "📁 Data Import",
        "🗺 Map",
        "W1: 8-Hour Volume",
        "W2: 4-Hour Volume",
        "W3: Peak Hour",
        "W4: Pedestrian",
        "W5: School Crossing",
        "W6: Coordinated",
        "W7: Crash",
        "W8: Network",
        "W9: Grade Crossing",
        "♿ Accessibility",
        "📊 Summary",
    ])

    with tabs[0]:  render_data_import_tab()
    with tabs[1]:  render_map_tab()
    with tabs[2]:  render_w1_tab()
    with tabs[3]:  render_w2_tab()
    with tabs[4]:  render_w3_tab()
    with tabs[5]:  render_w4_tab()
    with tabs[6]:  render_w5_tab()
    with tabs[7]:  render_w6_tab()
    with tabs[8]:  render_w7_tab()
    with tabs[9]:  render_w8_tab()
    with tabs[10]: render_w9_tab()
    with tabs[11]: render_accessibility_tab()
    with tabs[12]: render_summary_tab()


if __name__ == "__main__":
    main()
