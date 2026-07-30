"""
star_routes_exact.py
Exact waypoint coordinates (WGS84) and STAR route sequences for VOBL (Bengaluru).
Source: STAR Chart RWY 27L/27R – ADKAL 7P, GUNIM 7P, LEKAP 7P, PEXEG 7P,
        RIKBU 7P, SUSIK 7P, TELUV 7P, UGABA 7P  (RNAV1 GNSS)
"""

def dms(d, m, s):
    """Convert Degrees, Minutes, Seconds to decimal degrees."""
    return d + m / 60.0 + s / 3600.0


# ─────────────────────────────────────────────────────────────────────────────
# ALL WAYPOINTS  (name → (lat, lon))
# ─────────────────────────────────────────────────────────────────────────────
WAYPOINTS = {
    # Initial / En-route fixes
    'TELUV': (dms(14, 9, 42.00),  dms(78, 11, 13.00)),
    'RURPA': (dms(13, 39, 44.93), dms(77, 56, 52.57)),
    'XIVIL': (dms(13, 17, 39.68), dms(78, 55,  4.01)),
    'BEMSO': (dms(13, 15, 41.74), dms(78, 27, 54.39)),
    'LEKAP': (dms(12,  5,  5.30), dms(79, 35, 54.80)),
    'GURBI': (dms(12, 51, 54.53), dms(77, 52, 13.75)),
    'ADKAL': (dms(11, 50, 54.00), dms(78, 16, 49.00)),
    'UGABA': (dms(12,  8, 59.14), dms(77, 17, 29.41)),
    'MUGPA': (dms(12, 35, 13.13), dms(77, 28, 23.87)),
    'RUBOX': (dms(12, 53, 49.00), dms(77, 36,  9.31)),
    'PEXEG': (dms(13,  4, 15.10), dms(76,  2, 30.20)),
    'GUNIM': (dms(14,  5,  8.30), dms(75, 32,  8.50)),
    'OLNUR': (dms(13, 47,  3.12), dms(76,  6, 36.92)),
    'SUSIK': (dms(14, 24,  4.60), dms(76,  9, 56.30)),
    'GOMIL': (dms(14, 13,  4.78), dms(76, 58, 41.16)),
    'MAKAM': (dms(13, 49, 26.63), dms(76, 55, 34.73)),
    'RIKBU': (dms(15,  9, 57.00), dms(76, 21, 34.00)),
    'DUBEL': (dms(13, 18, 40.35), dms(77, 35, 47.05)),
    'APERU': (dms(13, 18, 38.22), dms(77, 40, 32.72)),   # MERGE POINT
    'ATPUK': (dms(13, 10, 14.15), dms(77, 16,  5.08)),
    'RIGBA': (dms(13, 35,  5.36), dms(77, 23, 22.01)),

    # Intermediate / Transition fixes
    'BL809': (dms(12, 43, 25.10), dms(78, 11, 11.33)),
    'BL810': (dms(12, 39, 13.40), dms(77, 58, 58.46)),
    'BL557': (dms(13,  6, 19.16), dms(77, 51, 45.88)),
    'BL558': (dms(13,  6, 18.35), dms(77, 59, 15.54)),   # corrected to 77°
    'BL556': (dms(13,  6, 23.46), dms(77, 41, 25.33)),
    'BL807': (dms(13,  7, 25.62), dms(76, 41,  0.20)),
    'BL806': (dms(13, 27, 50.84), dms(76, 42, 58.39)),
    'BL805': (dms(14,  0, 59.73), dms(77,  6, 33.26)),
    'BL704': (dms(13, 47, 44.47), dms(77, 15,  9.92)),

    # Point Merge Arc — OUTER (6000 ft AMSL)  arc goes West→North→East→South
    'BL272': (dms(13, 18, 33.14), dms(77, 51,  5.36)),
    'BL274': (dms(13, 24, 11.46), dms(78,  4, 38.84)),
    'BL276': (dms(13, 24,  7.11), dms(78, 11, 40.81)),
    'BL278': (dms(13, 20, 21.37), dms(78, 14, 14.30)),
    'BL280': (dms(13, 16,  5.58), dms(78, 15, 46.13)),
    'BL282': (dms(13, 11, 35.41), dms(78, 16, 10.08)),
    'BL284': (dms(13,  8, 24.98), dms(78, 15, 37.92)),

    # Point Merge Arc — INNER (5500 ft AMSL)
    'BL270': (dms(13, 12, 17.72), dms(77, 57, 43.76)),
    'BL285': (dms(13, 16, 11.85), dms(78,  3, 41.97)),
    'BL283': (dms(13, 23,  9.58), dms(78, 14, 22.10)),
    'BL281': (dms(13, 19, 42.78), dms(78, 16, 14.11)),
    'BL279': (dms(13, 15, 58.38), dms(78, 17, 21.99)),
    'BL277': (dms(13, 12,  5.25), dms(78, 17, 43.10)),
    'BL275': (dms(13,  8, 12.66), dms(78, 17, 16.64)),
    'BL273': (dms(13,  4, 54.36), dms(78, 16, 14.26)),
    'BL271': (dms(13,  4, 55.25), dms(78,  7, 24.31)),
    'BL286': (dms(13, 10, 46.79), dms(78,  4, 44.18)),
}


# ─────────────────────────────────────────────────────────────────────────────
# STAR ROUTE SEQUENCES
# Each list = ordered waypoints from initial fix → Merge Point (APERU)
# Based on VOBL STAR RWY 27L/27R chart
# ─────────────────────────────────────────────────────────────────────────────
STAR_SEQUENCES = {
    # From SOUTH-EAST — via ADKAL
    'ADKAL_7P': [
        'ADKAL', 'BL809', 'GURBI', 'BL558', 'BL557',
        'BL556', 'RUBOX', 'DUBEL', 'APERU'
    ],

    # From FAR SOUTH-EAST — via LEKAP / XIVIL arc
    'LEKAP_7P': [
        'LEKAP', 'XIVIL', 'BEMSO',
        'BL284', 'BL282', 'BL280', 'BL278', 'BL276', 'BL274', 'BL272',
        'APERU'
    ],

    # From NORTH-EAST — via TELUV, joins outer arc
    'TELUV_7P': [
        'TELUV', 'RURPA', 'BL276', 'BL274', 'BL272',
        'APERU'
    ],

    # From WEST — via PEXEG direct
    'PEXEG_7P': [
        'PEXEG', 'BL807', 'ATPUK', 'BL556', 'DUBEL', 'APERU'
    ],

    # From NORTH-WEST — via GUNIM / OLNUR
    'GUNIM_7P': [
        'GUNIM', 'OLNUR', 'BL806', 'MAKAM', 'RIGBA', 'BL704',
        'BL805', 'ATPUK', 'BL556', 'DUBEL', 'APERU'
    ],

    # From NORTH — via SUSIK / GOMIL
    'SUSIK_7P': [
        'SUSIK', 'GOMIL', 'BL805', 'BL704', 'RIGBA',
        'ATPUK', 'BL556', 'DUBEL', 'APERU'
    ],

    # From FAR NORTH — via RIKBU, joins GUNIM path
    'RIKBU_7P': [
        'RIKBU', 'MAKAM', 'GOMIL', 'BL805', 'BL704', 'RIGBA',
        'ATPUK', 'BL556', 'DUBEL', 'APERU'
    ],

    # From SOUTH-WEST — via UGABA / MUGPA
    'UGABA_7P': [
        'UGABA', 'MUGPA', 'RUBOX', 'DUBEL', 'APERU'
    ],
}

# The two Point Merge sequencing arcs (horizontal legs)
OUTER_ARC = ['BL272', 'BL274', 'BL276', 'BL278', 'BL280', 'BL282', 'BL284']
INNER_ARC = ['BL270', 'BL285', 'BL283', 'BL281', 'BL279', 'BL277', 'BL275', 'BL273', 'BL271', 'BL286']

# STAR color palette for visual distinction on map
STAR_COLORS = {
    'ADKAL_7P': '#FF6B6B',    # Coral Red
    'LEKAP_7P': '#FFD93D',    # Yellow
    'TELUV_7P': '#6BCB77',    # Green
    'PEXEG_7P': '#4D96FF',    # Blue
    'GUNIM_7P': '#FF922B',    # Orange
    'SUSIK_7P': '#CC5DE8',    # Purple
    'RIKBU_7P': '#F06595',    # Pink
    'UGABA_7P': '#20C997',    # Teal
}


def get_route_coords(star_name):
    """Return list of (lat, lon) tuples for a given STAR name."""
    seq = STAR_SEQUENCES.get(star_name, [])
    return [WAYPOINTS[wp] for wp in seq if wp in WAYPOINTS]


def get_arc_coords(arc_type='outer'):
    """Return coords for the Point Merge Arc."""
    arc = OUTER_ARC if arc_type == 'outer' else INNER_ARC
    return [WAYPOINTS[wp] for wp in arc if wp in WAYPOINTS]
