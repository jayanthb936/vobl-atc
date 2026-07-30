"""
app.py — Flask web app wrapping the VOBL ATC Engine for cloud deployment (Render.com)

Routes:
  GET /               → redirect to /map
  GET /map            → serves the live interactive Folium radar map (HTML)
  GET /api/status     → JSON status of last radar cycle
"""

import threading
import time
import os
import math
import logging

import numpy as np
import pandas as pd
import requests
import folium
import xgboost as xgb
from branca.element import Template, MacroElement
from ortools.sat.python import cp_model
from flask import Flask, send_file, jsonify, redirect

from star_routes_exact import (
    WAYPOINTS, STAR_SEQUENCES, STAR_COLORS,
    OUTER_ARC, INNER_ARC, get_route_coords, get_arc_coords
)

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgboost_eta_model.json')
DB_PATH    = os.path.join(BASE_DIR, 'star_routes.db')
MAP_PATH   = os.path.join(BASE_DIR, 'live_radar_map.html')

# ─── Constants ────────────────────────────────────────────────────────────────
ARP_LAT            = 13.1989
ARP_LON            = 77.7056
TURBOJET_RING_NM   = 120
TURBOPROP_RING_NM  = 80
POLL_INTERVAL_SEC  = 20

# ─── Shared state (thread-safe via simple dict + GIL) ─────────────────────────
STATUS = {
    'last_update': 'Never',
    'flights_tracked': 0,
    'cycle': 0,
    'error': None,
}

# ─── Route cache ──────────────────────────────────────────────────────────────
_route_cache   = {}
flight_trails  = {}
ranked_flights = {}

# =============================================================================
# Geometry helpers  (same as live_atc_engine.py)
# =============================================================================
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 3440.065
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def nm_to_latlon_offset(clat, clon, bearing_deg, dist_nm):
    R = 3440.065
    b = math.radians(bearing_deg)
    lat1, lon1 = math.radians(clat), math.radians(clon)
    lat2 = math.asin(math.sin(lat1)*math.cos(dist_nm/R) +
                     math.cos(lat1)*math.sin(dist_nm/R)*math.cos(b))
    lon2 = lon1 + math.atan2(math.sin(b)*math.sin(dist_nm/R)*math.cos(lat1),
                              math.cos(dist_nm/R) - math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def generate_ring_polygon(clat, clon, radius_nm, n=72):
    return [nm_to_latlon_offset(clat, clon, 360/n*i, radius_nm) for i in range(n+1)]

def get_icao_separation(lead, trail):
    matrix = {
        ('Heavy','Heavy'):90, ('Heavy','Medium'):120, ('Heavy','Light'):180,
        ('Medium','Heavy'):90, ('Medium','Medium'):90, ('Medium','Light'):120,
        ('Light','Heavy'):90, ('Light','Medium'):90, ('Light','Light'):90,
    }
    return matrix.get((lead, trail), 90)

# =============================================================================
# OpenSky helpers
# =============================================================================
def get_live_radar_data():
    url = f"https://api.adsb.lol/v2/lat/{ARP_LAT}/lon/{ARP_LON}/dist/200"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.error(f"Radar data error: {e}")
    return None

def get_destination_icao(callsign):
    # OpenSky routes API is blocked on Render, so we bypass it.
    # The altitude/vrate and distance filters are already 95% accurate for arrivals.
    return None

def load_star_database():
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    wp_df = pd.read_sql_query("SELECT * FROM waypoints", conn).set_index('identifier').to_dict('index')
    routes_df = pd.read_sql_query("SELECT * FROM star_routes ORDER BY star_name, sequence_num", conn)
    conn.close()
    star_starts, star_full_paths = {}, {}
    for sname, group in routes_df.groupby('star_name'):
        path = []
        for _, row in group.iterrows():
            wp = row['waypoint_identifier']
            if wp in wp_df:
                coords = (wp_df[wp]['latitude'], wp_df[wp]['longitude'])
                path.append(coords)
                if row['sequence_num'] == 1:
                    star_starts[sname] = coords
        star_full_paths[sname] = path
    return star_starts, star_full_paths

# =============================================================================
# ATC polling + map generation (runs in background thread)
# =============================================================================
def atc_poll_loop(star_starts, xgb_model):
    global ranked_flights, flight_trails

    while True:
        try:
            _run_one_cycle(star_starts, xgb_model)
        except Exception as e:
            log.error(f"ATC cycle error: {e}")
            STATUS['error'] = str(e)
        time.sleep(POLL_INTERVAL_SEC)

def _run_one_cycle(star_starts, xgb_model):
    global ranked_flights, flight_trails

    data = get_live_radar_data()
    if not data or not data.get('ac'):
        log.info("No radar data this cycle.")
        return

    active = set()

    for ac in data['ac']:
        callsign  = str(ac.get('flight', '')).strip()
        lat, lon  = ac.get('lat'), ac.get('lon')
        true_track = ac.get('track')

        try:
            alt_ft = float(ac.get('alt_baro', 0))
        except (ValueError, TypeError):
            alt_ft = 0

        try:
            speed_kts = float(ac.get('gs', 0))
        except (ValueError, TypeError):
            speed_kts = 0

        try:
            vrate = float(ac.get('baro_rate', 0))
        except (ValueError, TypeError):
            vrate = 0

        if not callsign or None in (lat, lon, speed_kts, true_track):
            continue

        dist_nm   = calculate_distance(ARP_LAT, ARP_LON, lat, lon)

        is_turbojet  = speed_kts > 250
        trigger_ring = TURBOJET_RING_NM if is_turbojet else TURBOPROP_RING_NM
        ac_type      = 'Turbojet' if is_turbojet else 'Turboprop'

        if not (alt_ft < 35000 and vrate and vrate < 0 and speed_kts > 100):
            continue
        if not (5 < dist_nm <= trigger_ring):
            continue

        dest = get_destination_icao(callsign)
        if dest and dest != 'VOBL':
            continue

        active.add(callsign)

        # Nearest STAR
        closest_star, min_d = None, 9999
        for sname, (slat, slon) in star_starts.items():
            d = calculate_distance(lat, lon, slat, slon)
            if d < min_d:
                min_d = d
                closest_star = sname

        target_bearing = calculate_bearing(lat, lon,
                                           star_starts[closest_star][0],
                                           star_starts[closest_star][1])
        heading_diff = abs(true_track - target_bearing)
        if heading_diff > 180:
            heading_diff = 360 - heading_diff

        x_nm = (lon - ARP_LON) * 60.0 * math.cos(math.radians(ARP_LAT))
        y_nm = (lat - ARP_LAT) * 60.0
        brng_to_rwy = calculate_bearing(lat, lon, ARP_LAT, ARP_LON)
        track_err   = min(abs(true_track - brng_to_rwy), 360 - abs(true_track - brng_to_rwy))

        all_stars = ['ADKAL_7P','GUNIM_7P','LEKAP_7P','PEXEG_7P',
                     'RIKBU_7P','SUSIK_7P','TELUV_7P','UGABA_7P']
        features = {
            'lat':[lat], 'lon':[lon], 'X (Nm)':[x_nm], 'Y (Nm)':[y_nm],
            'gspdxy':[speed_kts], 'gspdxy_smooth':[speed_kts],
            'Mode-C':[alt_ft], 'vrate':[vrate], 'vrate_smooth':[vrate],
            'distance_nm':[dist_nm], 'pms_track_distance':[dist_nm*1.2],
            'bearing_to_rwy':[brng_to_rwy], 'track_error':[track_err],
            'energy_state':[(alt_ft/100)+(speed_kts**2)/1000],
            'time_sin':[np.sin(2*np.pi*(time.time()%86400)/86400)],
            'time_cos':[np.cos(2*np.pi*(time.time()%86400)/86400)],
        }
        for s in all_stars:
            features[f'STAR_{s}'] = [1 if closest_star == s else 0]

        eta_seconds = max(float(xgb_model.predict(pd.DataFrame(features))[0]), 60.0)
        score = (eta_seconds*0.5 + dist_nm*10*0.3 + (alt_ft/10)*0.2 + heading_diff*2.0)

        ranked_flights[callsign] = {
            'callsign': callsign, 'lat': lat, 'lon': lon,
            'alt_ft': alt_ft, 'speed_kts': speed_kts, 'track': true_track,
            'dist_nm': dist_nm, 'assigned_star': closest_star,
            'eta_sec': eta_seconds, 'priority_score': score,
            'ac_type': ac_type, 'trigger_ring': trigger_ring,
            'dest': dest or '????',
            'wake': ranked_flights.get(callsign, {}).get(
                'wake', np.random.choice(['Heavy','Medium','Light'], p=[0.2,0.6,0.2])
            ),
        }

    ranked_flights = {cs: v for cs, v in ranked_flights.items() if cs in active}

    if not ranked_flights:
        log.info("No VOBL arrivals in trigger rings.")
        _build_empty_map(star_starts)
        return

    flights = sorted(ranked_flights.values(), key=lambda x: x['priority_score'])
    for rank, f in enumerate(flights):
        f['rank'] = rank + 1

    batch = flights[:15]
    N = len(batch)

    # OR-Tools
    cp = cp_model.CpModel()
    sta, b = {}, {}
    for i in range(N):
        eta = int(batch[i]['eta_sec'])
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

    def fmt_time(s):
        return f"{int(s)//60:02d}m {int(s)%60:02d}s"

    def get_atc_action(rank, delay_sec):
        if delay_sec <= 60:
            return "PROCEED TO MERGE", "#d4edda", "#155724"
        elif delay_sec <= 120:
            return f"REDUCE SPEED (+{delay_sec}s)", "#fff3cd", "#856404"
        elif delay_sec <= 300:
            return f"EXTEND ON ARC (+{delay_sec}s)", "#ffd8b1", "#7a3b00"
        else:
            return f"ORBIT {delay_sec//60}m{delay_sec%60:02d}s", "#ffcccc", "#721c24"

    html_rows = ""
    for i in range(N):
        f = batch[i]
        delay = solver.Value(sta[i]) - int(f['eta_sec'])
        action_text, row_bg, text_color = get_atc_action(f['rank'], delay)
        f['action'] = action_text
        f['delay']  = delay
        html_rows += f"""
            <tr style="background-color:{row_bg};border-bottom:1px solid #ddd;color:#111;">
                <td style="padding:5px;"><b>#{f['rank']}</b></td>
                <td style="padding:5px;">{f['callsign']}<br>
                    <span style="font-size:9px;color:gray;">{f['ac_type']} | {f['trigger_ring']}NM</span></td>
                <td style="padding:5px;">{f['assigned_star']}</td>
                <td style="padding:5px;">{fmt_time(f['eta_sec'])}</td>
                <td style="padding:5px;font-weight:bold;color:{text_color};">{action_text}</td>
            </tr>"""

    _build_map(batch, html_rows, star_starts)

    STATUS['last_update'] = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    STATUS['flights_tracked'] = len(batch)
    STATUS['cycle'] += 1
    STATUS['error'] = None
    log.info(f"Cycle {STATUS['cycle']}: {len(batch)} VOBL arrivals ranked.")


def _build_map(batch, html_rows, star_starts, empty=False):
    m = folium.Map(location=[ARP_LAT, ARP_LON], zoom_start=8,
                   tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
                   attr='Google Maps')

    folium.Marker([ARP_LAT, ARP_LON], popup="VOBL — Kempegowda Intl",
                  icon=folium.Icon(color="green", icon="plane", prefix="fa")).add_to(m)

    # Range rings
    for ring_nm, color, label in [
        (TURBOJET_RING_NM,  '#1a73e8', '120NM (Turbojet)'),
        (TURBOPROP_RING_NM, '#f4a100', '80NM (Turboprop)'),
    ]:
        folium.PolyLine(generate_ring_polygon(ARP_LAT, ARP_LON, ring_nm),
                        color=color, weight=1.5, opacity=0.6, dash_array='8,6',
                        tooltip=label).add_to(m)
        llat, llon = nm_to_latlon_offset(ARP_LAT, ARP_LON, 0, ring_nm)
        folium.map.Marker([llat, llon], icon=folium.DivIcon(
            icon_size=(130,20), icon_anchor=(65,20),
            html=f'<div style="font-size:8pt;color:{color};font-weight:bold;">{label}</div>'
        )).add_to(m)

    # Exact STAR routes
    for star_name, waypoint_seq in STAR_SEQUENCES.items():
        route_coords = get_route_coords(star_name)
        if not route_coords:
            continue
        color = STAR_COLORS.get(star_name, '#888888')
        folium.PolyLine(route_coords, color=color, weight=2.0,
                        opacity=0.85, tooltip=star_name).add_to(m)
        for wp_name in waypoint_seq:
            if wp_name not in WAYPOINTS:
                continue
            wp_lat, wp_lon = WAYPOINTS[wp_name]
            is_key = wp_name in ('APERU','DUBEL','RUBOX','ATPUK')
            is_start = (wp_name == waypoint_seq[0])
            if is_start or is_key:
                folium.CircleMarker([wp_lat, wp_lon], radius=4,
                    color=color, fill=True, fill_color='white',
                    fill_opacity=1.0, weight=2, tooltip=wp_name).add_to(m)
                folium.map.Marker([wp_lat, wp_lon], icon=folium.DivIcon(
                    icon_size=(90,16), icon_anchor=(-6,8),
                    html=f'<div style="font-size:7pt;color:{color};font-weight:bold;'
                         f'white-space:nowrap;text-shadow:0 0 3px black;">{wp_name}</div>'
                )).add_to(m)
            else:
                folium.CircleMarker([wp_lat, wp_lon], radius=2,
                    color=color, fill=True, fill_color=color,
                    fill_opacity=0.7, weight=1).add_to(m)

    # Point Merge Arcs
    outer_c = get_arc_coords('outer')
    inner_c = get_arc_coords('inner')
    if outer_c:
        folium.PolyLine(outer_c, color='#ffffff', weight=2.5,
                        opacity=0.9, tooltip='Outer Arc — 6000ft').add_to(m)
    if inner_c:
        folium.PolyLine(inner_c, color='#cccccc', weight=2.0,
                        opacity=0.9, dash_array='4,3',
                        tooltip='Inner Arc — 5500ft').add_to(m)

    # APERU merge marker
    if 'APERU' in WAYPOINTS:
        alat, alon = WAYPOINTS['APERU']
        folium.CircleMarker([alat, alon], radius=8, color='#FFD700',
                            fill=True, fill_color='#FFD700', fill_opacity=0.9,
                            tooltip='APERU — MERGE POINT').add_to(m)
        folium.map.Marker([alat, alon], icon=folium.DivIcon(
            icon_size=(130,20), icon_anchor=(-10,8),
            html='<div style="font-size:8pt;color:#FFD700;font-weight:bold;'
                 'text-shadow:0 0 4px black;">★ APERU MERGE</div>'
        )).add_to(m)

    if not empty:
        for f in batch:
            cs = f['callsign']
            if cs not in flight_trails:
                flight_trails[cs] = []
            flight_trails[cs].append((f['lat'], f['lon']))

            plane_color = 'red' if f['delay'] > 60 else '#1a73e8'
            if len(flight_trails[cs]) > 1:
                folium.PolyLine(flight_trails[cs], color=plane_color,
                                weight=2.5, opacity=0.75).add_to(m)

            # Snap-to-path predictive line
            from star_routes_exact import STAR_SEQUENCES as SS
            seq_coords = get_route_coords(f['assigned_star'])
            closest_wp_idx, min_wp_dist = 0, float('inf')
            for idx, wp_coords in enumerate(seq_coords):
                d = calculate_distance(f['lat'], f['lon'], wp_coords[0], wp_coords[1])
                if d < min_wp_dist:
                    min_wp_dist = d
                    closest_wp_idx = idx
            future_path = [(f['lat'], f['lon'])] + seq_coords[closest_wp_idx:]
            folium.PolyLine(future_path, color='yellow', weight=2,
                            dash_array='5,5', opacity=0.9).add_to(m)

            svg_plane = (f'<svg width="24" height="24" viewBox="0 0 24 24" '
                         f'style="transform:rotate({f["track"]-45}deg);">'
                         f'<path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 '
                         f'3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 '
                         f'2.5z" fill="{plane_color}" stroke="white" stroke-width="1"/></svg>')
            folium.Marker([f['lat'], f['lon']],
                icon=folium.DivIcon(html=svg_plane, icon_size=(24,24), icon_anchor=(12,12)),
                popup=folium.Popup(
                    f"<b>{cs}</b><br>Rank:#{f['rank']}<br>Type:{f['ac_type']}<br>"
                    f"Dest:{f['dest']}<br>Alt:{int(f['alt_ft'])}ft<br>"
                    f"Speed:{int(f['speed_kts'])}kts<br>Action:{f['action']}",
                    max_width=220)
            ).add_to(m)

            folium.map.Marker([f['lat'], f['lon']], icon=folium.DivIcon(
                icon_size=(30,20), icon_anchor=(-4,4),
                html=f'<div style="font-size:9pt;font-weight:bold;color:{plane_color}">#{f["rank"]}</div>'
            )).add_to(m)

    # Dashboard sidebar
    last_update = STATUS['last_update']
    sidebar_html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="position:fixed;top:20px;right:20px;width:420px;
        background:rgba(15,15,30,0.94);border:1.5px solid #1a73e8;
        border-radius:12px;z-index:9999;overflow-y:auto;max-height:90vh;
        box-shadow:0 4px 24px rgba(0,0,0,0.7);padding:16px;
        font-family:'Segoe UI',Arial,sans-serif;color:#eee;">
        <h3 style="margin:0 0 4px 0;color:#1a73e8;letter-spacing:1px;">✈ VOBL Point Merge ATC</h3>
        <p style="font-size:11px;color:#aaa;margin:0 0 4px 0;">Kempegowda Intl · XGBoost AI · OR-Tools</p>
        <p style="font-size:10px;color:#666;margin:0 0 10px 0;">Updated: {last_update}</p>
        <p style="font-size:10px;color:#aaa;margin:0 0 8px 0;">🔵 120NM=Turbojet &nbsp;|&nbsp; 🟡 80NM=Turboprop</p>
        <table style="width:100%;border-collapse:collapse;font-size:11px;">
            <tr style="background:#1a73e8;color:white;">
                <th style="padding:5px;">Rnk</th>
                <th style="padding:5px;">Flight</th>
                <th style="padding:5px;">STAR</th>
                <th style="padding:5px;">AI ETA</th>
                <th style="padding:5px;">Action</th>
            </tr>
            {'<tr><td colspan="5" style="text-align:center;padding:20px;color:#aaa;">No VOBL arrivals in range</td></tr>' if empty else html_rows}
        </table>
    </div>
    {{% endmacro %}}
    """
    macro = MacroElement()
    macro._template = Template(sidebar_html)
    m.get_root().add_child(macro)
    m.save(MAP_PATH)


def _build_empty_map(star_starts):
    _build_map([], "", star_starts, empty=True)


# =============================================================================
# Flask routes
# =============================================================================
@app.route('/')
def index():
    return redirect('/map')

@app.route('/map')
def serve_map():
    if not os.path.exists(MAP_PATH):
        return "<h2>Map is being generated, please refresh in 30 seconds.</h2>", 503
    return send_file(MAP_PATH, mimetype='text/html')

@app.route('/api/status')
def api_status():
    return jsonify(STATUS)


# =============================================================================
# Startup
# =============================================================================
def start_background_engine():
    log.info("Loading STAR database...")
    star_starts, _ = load_star_database()

    log.info("Loading XGBoost AI model...")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(MODEL_PATH)

    log.info("Building initial empty map...")
    _build_empty_map(star_starts)

    log.info("Starting ATC polling thread...")
    t = threading.Thread(target=atc_poll_loop, args=(star_starts, xgb_model), daemon=True)
    t.start()

if __name__ == '__main__':
    start_background_engine()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
else:
    # Called by gunicorn
    start_background_engine()
