import sqlite3
import requests
import math
import time
import os
import pandas as pd
from ortools.sat.python import cp_model
import numpy as np
import folium
from branca.element import Template, MacroElement
import xgboost as xgb
from star_routes_exact import (
    WAYPOINTS, STAR_SEQUENCES, STAR_COLORS,
    OUTER_ARC, INNER_ARC, get_route_coords, get_arc_coords
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ARP_LAT = 13.1989
ARP_LON = 77.7056

TURBOJET_RING_NM  = 120   # Turbojets get ranked when they enter this ring
TURBOPROP_RING_NM = 80    # Turboprops get ranked when they enter this ring

# Route cache: callsign → destination ICAO. Prevents hammering the API.
_route_cache = {}

flight_trails = {}      # callsign → [(lat,lon), ...]
ranked_flights = {}     # callsign → dict   (persists rank across cycles)


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine distance in Nautical Miles."""
    R = 3440.065
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon**2/4))
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    """True bearing from point 1 to point 2 (degrees)."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def nm_to_latlon_offset(center_lat, center_lon, bearing_deg, distance_nm):
    """Return a lat/lon point at <distance_nm> NM from center in <bearing_deg> direction."""
    R = 3440.065
    brng = math.radians(bearing_deg)
    lat1 = math.radians(center_lat)
    lon1 = math.radians(center_lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(distance_nm/R) +
                     math.cos(lat1)*math.sin(distance_nm/R)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(distance_nm/R)*math.cos(lat1),
                              math.cos(distance_nm/R) - math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def generate_ring_polygon(center_lat, center_lon, radius_nm, num_points=72):
    """Generate a list of (lat, lon) forming a circle of <radius_nm> around center."""
    pts = []
    for i in range(num_points + 1):
        bearing = (360 / num_points) * i
        pts.append(nm_to_latlon_offset(center_lat, center_lon, bearing, radius_nm))
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────
def load_star_database(db_path):
    conn = sqlite3.connect(db_path)
    waypoints = pd.read_sql_query("SELECT * FROM waypoints", conn).set_index('identifier').to_dict('index')
    routes_df = pd.read_sql_query("SELECT * FROM star_routes ORDER BY star_name, sequence_num", conn)
    conn.close()

    star_start_coords = {}
    star_full_paths   = {}

    for star_name, group in routes_df.groupby('star_name'):
        path = []
        for _, row in group.iterrows():
            wp = row['waypoint_identifier']
            if wp in waypoints:
                coords = (waypoints[wp]['latitude'], waypoints[wp]['longitude'])
                path.append(coords)
                if row['sequence_num'] == 1:
                    star_start_coords[star_name] = coords
        star_full_paths[star_name] = path

    return star_start_coords, star_full_paths


# ─────────────────────────────────────────────────────────────────────────────
# OPENSKY APIs
# ─────────────────────────────────────────────────────────────────────────────
def get_live_opensky_data():
    url = "https://opensky-network.org/api/states/all?lamin=11.5&lomin=76.0&lamax=15.0&lomax=79.5"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"OpenSky states error: {e}")
    return None

def get_destination_icao(callsign):
    """
    Query the OpenSky /routes endpoint to get the filed destination for a callsign.
    Results are cached so we don't re-query every 20 s.
    Returns destination ICAO string or None.
    """
    if callsign in _route_cache:
        return _route_cache[callsign]

    try:
        url = f"https://opensky-network.org/api/routes?callsign={callsign}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            data = r.json()
            route = data.get('route', [])
            # route is e.g. ["VIDP", "VOBL"] — last element is destination
            dest = route[-1] if route else None
            _route_cache[callsign] = dest
            return dest
        else:
            _route_cache[callsign] = None   # cache miss to avoid repeated calls
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ICAO WAKE SEPARATION
# ─────────────────────────────────────────────────────────────────────────────
def get_icao_separation(lead, trail):
    matrix = {
        ('Heavy',  'Heavy' ): 90,
        ('Heavy',  'Medium'): 120,
        ('Heavy',  'Light' ): 180,
        ('Medium', 'Heavy' ): 90,
        ('Medium', 'Medium'): 90,
        ('Medium', 'Light' ): 120,
        ('Light',  'Heavy' ): 90,
        ('Light',  'Medium'): 90,
        ('Light',  'Light' ): 90,
    }
    return matrix.get((lead, trail), 90)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ATC CYCLE
# ─────────────────────────────────────────────────────────────────────────────
def run_atc_cycle(star_starts, star_paths, xgb_model):
    global ranked_flights

    print("\nPolling Live OpenSky Radar Data...")
    data = get_live_opensky_data()

    if not data or not data.get('states'):
        print("No flights found or API rate limit reached.")
        return

    active_callsigns = set()   # track who is still in the air this cycle

    for state in data['states']:
        callsign  = str(state[1]).strip()
        lon       = state[5]
        lat       = state[6]
        baro_alt  = state[7]
        velocity  = state[9]
        true_track = state[10]
        vrate     = state[11]

        if not callsign or lat is None or lon is None or velocity is None or true_track is None:
            continue

        alt_ft    = baro_alt * 3.28084 if baro_alt else 0
        speed_kts = velocity * 1.94384
        dist_nm   = calculate_distance(ARP_LAT, ARP_LON, lat, lon)

        # ── Classify aircraft type by speed ──────────────────────────────────
        is_turbojet  = speed_kts > 250
        trigger_ring = TURBOJET_RING_NM if is_turbojet else TURBOPROP_RING_NM
        ac_type      = 'Turbojet' if is_turbojet else 'Turboprop'

        # ── Basic sanity filters (descending commercial traffic) ──────────────
        if not (alt_ft < 35000 and vrate and vrate < 0 and speed_kts > 100):
            continue
        if not (5 < dist_nm <= trigger_ring):
            continue

        # ── DESTINATION CHECK via OpenSky /routes ────────────────────────────
        dest = get_destination_icao(callsign)
        if dest and dest != 'VOBL':
            # Filed destination is NOT Bangalore — skip silently
            continue
        # If dest is None (no route data), we fall through and still process.
        # The ring + descent filters are good enough for that edge case.

        # ── Already ranked? Update position only. ────────────────────────────
        active_callsigns.add(callsign)

        # ── Assign nearest STAR ───────────────────────────────────────────────
        closest_star, min_d = None, 9999
        for sname, (slat, slon) in star_starts.items():
            d = calculate_distance(lat, lon, slat, slon)
            if d < min_d:
                min_d = d
                closest_star = sname

        # ── Heading alignment score ───────────────────────────────────────────
        target_bearing = calculate_bearing(lat, lon,
                                           star_starts[closest_star][0],
                                           star_starts[closest_star][1])
        heading_diff = abs(true_track - target_bearing)
        if heading_diff > 180:
            heading_diff = 360 - heading_diff

        # ── Build XGBoost feature vector ─────────────────────────────────────
        x_nm = (lon - ARP_LON) * 60.0 * math.cos(math.radians(ARP_LAT))
        y_nm = (lat - ARP_LAT) * 60.0
        brng_to_rwy = calculate_bearing(lat, lon, ARP_LAT, ARP_LON)
        track_err   = min(abs(true_track - brng_to_rwy), 360 - abs(true_track - brng_to_rwy))

        features = {
            'lat': [lat], 'lon': [lon],
            'X (Nm)': [x_nm], 'Y (Nm)': [y_nm],
            'gspdxy': [speed_kts], 'gspdxy_smooth': [speed_kts],
            'Mode-C': [alt_ft],
            'vrate': [vrate], 'vrate_smooth': [vrate],
            'distance_nm': [dist_nm],
            'pms_track_distance': [dist_nm * 1.2],
            'bearing_to_rwy': [brng_to_rwy],
            'track_error': [track_err],
            'energy_state': [(alt_ft / 100) + (speed_kts**2) / 1000],
            'time_sin': [np.sin(2 * np.pi * (time.time() % 86400) / 86400)],
            'time_cos': [np.cos(2 * np.pi * (time.time() % 86400) / 86400)],
        }
        all_stars = ['ADKAL_7P','GUNIM_7P','LEKAP_7P','PEXEG_7P',
                     'RIKBU_7P','SUSIK_7P','TELUV_7P','UGABA_7P']
        for s in all_stars:
            features[f'STAR_{s}'] = [1 if closest_star == s else 0]

        feat_df      = pd.DataFrame(features)
        eta_seconds  = max(float(xgb_model.predict(feat_df)[0]), 60.0)

        score = (eta_seconds * 0.5 +
                 dist_nm * 10 * 0.3 +
                 (alt_ft / 10) * 0.2 +
                 heading_diff * 2.0)

        ranked_flights[callsign] = {
            'callsign':      callsign,
            'lat':           lat,
            'lon':           lon,
            'alt_ft':        alt_ft,
            'speed_kts':     speed_kts,
            'track':         true_track,
            'dist_nm':       dist_nm,
            'assigned_star': closest_star,
            'eta_sec':       eta_seconds,
            'priority_score': score,
            'ac_type':       ac_type,
            'trigger_ring':  trigger_ring,
            'dest':          dest or '????',
            'wake':          ranked_flights.get(callsign, {}).get(
                                 'wake',
                                 np.random.choice(['Heavy','Medium','Light'], p=[0.2,0.6,0.2])
                             ),
        }

    # Remove flights that have disappeared from radar
    ranked_flights = {cs: v for cs, v in ranked_flights.items() if cs in active_callsigns}

    if not ranked_flights:
        print("No VOBL-destined flights currently within trigger rings.")
        return

    # ── Re-rank by priority score ─────────────────────────────────────────────
    flights = sorted(ranked_flights.values(), key=lambda x: x['priority_score'])
    for rank, f in enumerate(flights):
        f['rank'] = rank + 1

    batch = flights[:15]
    N     = len(batch)

    # ── OR-Tools Sequencer ────────────────────────────────────────────────────
    cp = cp_model.CpModel()
    sta, b = {}, {}
    for i in range(N):
        eta    = int(batch[i]['eta_sec'])
        sta[i] = cp.NewIntVar(eta, eta + 3600, f'sta_{i}')
    for i in range(N):
        for j in range(i+1, N):
            b[(i,j)] = cp.NewBoolVar(f'b_{i}_{j}')
    for i in range(N):
        for j in range(i+1, N):
            sep_ij = get_icao_separation(batch[i]['wake'], batch[j]['wake'])
            sep_ji = get_icao_separation(batch[j]['wake'], batch[i]['wake'])
            cp.Add(sta[j] >= sta[i] + sep_ij).OnlyEnforceIf(b[(i,j)])
            cp.Add(sta[i] >= sta[j] + sep_ji).OnlyEnforceIf(b[(i,j)].Not())
            if batch[i]['rank'] < batch[j]['rank']:
                cp.Add(b[(i,j)] == 1)
            else:
                cp.Add(b[(i,j)] == 0)
    cp.Minimize(sum(sta[i] for i in range(N)))
    solver = cp_model.CpSolver()
    solver.Solve(cp)

    # ── Build HTML table rows ─────────────────────────────────────────────────
    def fmt_time(s):
        return f"{int(s)//60:02d}m {int(s)%60:02d}s"

    def get_atc_action(rank, delay_sec):
        """
        Translate OR-Tools delay into realistic ATC phraseology.

        Rank #1 = leading aircraft — always gets PROCEED unless natural
        separation deficit exists.

        delay_sec = STA − AI-ETA (the extra time OR-Tools needs to absorb
        to maintain ICAO wake separation ahead of the preceding aircraft).

        ≤ 60 s  → PROCEED TO MERGE   (natural spacing is sufficient)
        61–120 s → REDUCE SPEED       (pilot reduces speed ~20 kts on arc)
        121–300 s → EXTEND ON ARC     (fly further around the merge arc)
        > 300 s  → HOLD / ORBIT       (full holding pattern needed)
        """
        if rank == 1 and delay_sec <= 60:
            return "PROCEED TO MERGE", "#d4edda", "#155724"   # green
        if delay_sec <= 60:
            return "PROCEED TO MERGE", "#d4edda", "#155724"   # green
        elif delay_sec <= 120:
            return f"REDUCE SPEED  (+{delay_sec}s)", "#fff3cd", "#856404"   # amber
        elif delay_sec <= 300:
            return f"EXTEND ON ARC (+{delay_sec}s)", "#ffd8b1", "#7a3b00"   # orange
        else:
            mins = delay_sec // 60
            secs = delay_sec % 60
            return f"ORBIT {mins}m{secs:02d}s", "#ffcccc", "#721c24"         # red

    html_rows = ""
    for i in range(N):
        f       = batch[i]
        sta_val = solver.Value(sta[i])
        delay   = sta_val - int(f['eta_sec'])
        action_text, row_bg, text_color = get_atc_action(f['rank'], delay)
        f['action'] = action_text
        f['delay']  = delay
        html_rows += f"""
            <tr style="background-color:{row_bg}; border-bottom:1px solid #ddd; color:#111;">
                <td style="padding:5px;"><b>#{f['rank']}</b></td>
                <td style="padding:5px;">{f['callsign']}<br>
                    <span style="font-size:9px;color:gray;">{f['ac_type']} | {f['trigger_ring']}NM</span></td>
                <td style="padding:5px;">{f['assigned_star']}</td>
                <td style="padding:5px;">{fmt_time(f['eta_sec'])}</td>
                <td style="padding:5px;font-weight:bold;color:{text_color};">{action_text}</td>
            </tr>"""

    # ──────────────────────────────────────────────────────────────────────────
    # MAP GENERATION
    # ──────────────────────────────────────────────────────────────────────────
    print("Updating Google Maps Dashboard...")
    m = folium.Map(location=[ARP_LAT, ARP_LON], zoom_start=8,
                   tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
                   attr='Google Maps')

    # Airport marker
    folium.Marker([ARP_LAT, ARP_LON], popup="VOBL — Kempegowda Intl",
                  icon=folium.Icon(color="green", icon="plane", prefix="fa")).add_to(m)

    # ── 120 NM ring (Turbojet trigger) — dashed blue ──────────────────────────
    ring_120 = generate_ring_polygon(ARP_LAT, ARP_LON, TURBOJET_RING_NM)
    folium.PolyLine(ring_120, color='#1a73e8', weight=1.5,
                    opacity=0.6, dash_array='8, 6',
                    tooltip="120 NM — Turbojet Trigger").add_to(m)
    # Label
    label_120_lat, label_120_lon = nm_to_latlon_offset(ARP_LAT, ARP_LON, 0, TURBOJET_RING_NM)
    folium.map.Marker([label_120_lat, label_120_lon],
        icon=folium.DivIcon(icon_size=(100,20), icon_anchor=(50,20),
            html='<div style="font-size:8pt;color:#1a73e8;font-weight:bold;">120NM (Turbojet)</div>')
    ).add_to(m)

    # ── 80 NM ring (Turboprop trigger) — dashed orange ────────────────────────
    ring_80 = generate_ring_polygon(ARP_LAT, ARP_LON, TURBOPROP_RING_NM)
    folium.PolyLine(ring_80, color='#f4a100', weight=1.5,
                    opacity=0.6, dash_array='8, 6',
                    tooltip="80 NM — Turboprop Trigger").add_to(m)
    # Label
    label_80_lat, label_80_lon = nm_to_latlon_offset(ARP_LAT, ARP_LON, 0, TURBOPROP_RING_NM)
    folium.map.Marker([label_80_lat, label_80_lon],
        icon=folium.DivIcon(icon_size=(110,20), icon_anchor=(55,20),
            html='<div style="font-size:8pt;color:#f4a100;font-weight:bold;">80NM (Turboprop)</div>')
    ).add_to(m)

    # ── Exact STAR Routes from chart (color-coded) ────────────────────────────
    for star_name, waypoint_seq in STAR_SEQUENCES.items():
        route_coords = get_route_coords(star_name)
        if not route_coords:
            continue
        color = STAR_COLORS.get(star_name, '#888888')

        # Draw the route line
        folium.PolyLine(
            route_coords, color=color, weight=2.0, opacity=0.85,
            tooltip=star_name
        ).add_to(m)

        # Waypoint dots
        for wp_name in waypoint_seq:
            if wp_name not in WAYPOINTS:
                continue
            wp_lat, wp_lon = WAYPOINTS[wp_name]
            is_key = wp_name in ('APERU', 'DUBEL', 'RUBOX', 'ATPUK')
            is_start = (wp_name == waypoint_seq[0])

            if is_start or is_key:
                # Named circle marker for initial fix and key fixes
                folium.CircleMarker(
                    location=[wp_lat, wp_lon], radius=4,
                    color=color, fill=True, fill_color='white',
                    fill_opacity=1.0, weight=2,
                    popup=wp_name, tooltip=wp_name
                ).add_to(m)
                # Label
                folium.map.Marker([wp_lat, wp_lon],
                    icon=folium.DivIcon(
                        icon_size=(90, 16), icon_anchor=(-6, 8),
                        html=f'<div style="font-size:7pt;color:{color};'
                             f'font-weight:bold;white-space:nowrap;'
                             f'text-shadow:0px 0px 3px black;">{wp_name}</div>'
                    )
                ).add_to(m)
            else:
                # Small tick for intermediate waypoints
                folium.CircleMarker(
                    location=[wp_lat, wp_lon], radius=2,
                    color=color, fill=True, fill_color=color,
                    fill_opacity=0.7, weight=1
                ).add_to(m)

    # ── Point Merge Arcs (sequencing legs) ────────────────────────────────────
    outer_coords = get_arc_coords('outer')
    inner_coords = get_arc_coords('inner')

    if outer_coords:
        folium.PolyLine(outer_coords, color='#ffffff', weight=2.5,
                        opacity=0.9, tooltip='Outer Arc — 6000ft AMSL').add_to(m)
    if inner_coords:
        folium.PolyLine(inner_coords, color='#cccccc', weight=2.0,
                        opacity=0.9, dash_array='4,3',
                        tooltip='Inner Arc — 5500ft AMSL').add_to(m)

    # ── APERU merge point special marker ─────────────────────────────────────
    if 'APERU' in WAYPOINTS:
        aperu_lat, aperu_lon = WAYPOINTS['APERU']
        folium.CircleMarker(
            location=[aperu_lat, aperu_lon], radius=8,
            color='#FFD700', fill=True, fill_color='#FFD700',
            fill_opacity=0.9, weight=2,
            popup='APERU — MERGE POINT', tooltip='APERU — MERGE POINT'
        ).add_to(m)
        folium.map.Marker([aperu_lat, aperu_lon],
            icon=folium.DivIcon(
                icon_size=(130, 20), icon_anchor=(-10, 8),
                html='<div style="font-size:8pt;color:#FFD700;font-weight:bold;'
                     'text-shadow:0px 0px 4px black;">★ APERU MERGE</div>'
            )
        ).add_to(m)

    # ── Per-flight markers ────────────────────────────────────────────────────
    for f in batch:
        cs = f['callsign']
        if cs not in flight_trails:
            flight_trails[cs] = []
        flight_trails[cs].append((f['lat'], f['lon']))

        plane_color = 'red' if f['delay'] > 60 else '#1a73e8'

        # Past trail
        if len(flight_trails[cs]) > 1:
            folium.PolyLine(flight_trails[cs], color=plane_color,
                            weight=2.5, opacity=0.75).add_to(m)

        # Predictive snap-to-path
        assigned_path = star_paths[f['assigned_star']]
        closest_wp_idx, min_wp_dist = 0, float('inf')
        for idx, wp_coords in enumerate(assigned_path):
            d = calculate_distance(f['lat'], f['lon'], wp_coords[0], wp_coords[1])
            if d < min_wp_dist:
                min_wp_dist = d
                closest_wp_idx = idx
        future_path = [(f['lat'], f['lon'])] + assigned_path[closest_wp_idx:]
        folium.PolyLine(future_path, color='yellow', weight=2,
                        dash_array='5, 5', opacity=0.9).add_to(m)

        # Airplane SVG icon
        svg_plane = f"""
        <svg width="24" height="24" viewBox="0 0 24 24"
             style="transform:rotate({f['track']-45}deg);">
          <path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2
                   l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"
                fill="{plane_color}" stroke="white" stroke-width="1"/>
        </svg>"""
        icon = folium.DivIcon(html=svg_plane, icon_size=(24,24), icon_anchor=(12,12))

        popup_html = (f"<b>{cs}</b><br>Rank: #{f['rank']}<br>"
                      f"Type: {f['ac_type']}<br>Trigger: {f['trigger_ring']}NM<br>"
                      f"Dest: {f['dest']}<br>Alt: {int(f['alt_ft'])} ft<br>"
                      f"Speed: {int(f['speed_kts'])} kts<br>Action: {f['action']}")
        folium.Marker(location=[f['lat'], f['lon']], icon=icon,
                      popup=folium.Popup(popup_html, max_width=220)).add_to(m)

        # Rank badge
        folium.map.Marker([f['lat'], f['lon']],
            icon=folium.DivIcon(icon_size=(30,20), icon_anchor=(-4, 4),
                html=f'<div style="font-size:9pt;font-weight:bold;color:{plane_color}">#{f["rank"]}</div>')
        ).add_to(m)

    # ── Floating ATC Dashboard ────────────────────────────────────────────────
    sidebar_html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="
        position:fixed; top:20px; right:20px; width:420px;
        background:rgba(15,15,30,0.94); border:1.5px solid #1a73e8;
        border-radius:12px; z-index:9999; overflow-y:auto; max-height:90vh;
        box-shadow:0 4px 24px rgba(0,0,0,0.7); padding:16px;
        font-family:'Segoe UI',Arial,sans-serif; color:#eee;">

        <h3 style="margin:0 0 4px 0; color:#1a73e8; letter-spacing:1px;">
            ✈ VOBL Point Merge ATC
        </h3>
        <p style="font-size:11px;color:#aaa;margin:0 0 10px 0;">
            Kempegowda Intl · XGBoost AI · OR-Tools Active
        </p>
        <p style="font-size:10px;color:#aaa;margin:0 0 8px 0;">
            🔵 120NM ring = Turbojet trigger &nbsp;|&nbsp; 🟡 80NM ring = Turboprop trigger
        </p>

        <table style="width:100%;border-collapse:collapse;font-size:11px;">
            <tr style="background:#1a73e8;color:white;">
                <th style="padding:5px;">Rnk</th>
                <th style="padding:5px;">Flight</th>
                <th style="padding:5px;">STAR</th>
                <th style="padding:5px;">AI ETA</th>
                <th style="padding:5px;">Action</th>
            </tr>
            {html_rows}
        </table>
    </div>
    {{% endmacro %}}
    """
    macro = MacroElement()
    macro._template = Template(sidebar_html)
    m.get_root().add_child(macro)

    map_path = r"C:\Users\Jayanth B\PycharmProjects\PythonProject\live_radar_map.html"
    m.save(map_path)
    print(f"Map updated — {len(batch)} VOBL arrivals ranked.")
    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    db_file    = r"C:\Users\Jayanth B\PycharmProjects\PythonProject\star_routes.db"
    model_path = r"C:\Users\Jayanth B\PycharmProjects\PythonProject\models\xgboost_eta_model.json"

    if not os.path.exists(db_file):
        print("ERROR: star_routes.db not found.")
    elif not os.path.exists(model_path):
        print("ERROR: xgboost_eta_model.json not found. Run 02_model_training.py first.")
    else:
        print("Loading STAR database...")
        star_starts, star_paths = load_star_database(db_file)

        print("Loading XGBoost AI Model...")
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(model_path)

        print("Starting INFINITE Live Loop. Press Ctrl+C to stop.")
        while True:
            try:
                run_atc_cycle(star_starts, star_paths, xgb_model)
                print("Waiting 20 seconds for next radar sweep...")
                time.sleep(20)
            except KeyboardInterrupt:
                print("\nATC Engine Stopped by User.")
                break
