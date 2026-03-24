"""
Unitree L2 Rover — Obstacle Detection + Path Planning
======================================================
THREE THREADS — no freezing, no stopping:
    1. socket_receiver  — reads points from bridge continuously
    2. compute_worker   — APF + path planning in background
    3. main thread      — draws the map smoothly at fixed rate

START ORDER:
    Terminal 1:  python3 ~/lidar_rover.py
    Terminal 2:  cd ~/unilidar_sdk2/unitree_lidar_sdk/bin && ./lidar_bridge
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import threading
import socket
import struct
import time
import collections

# ─── TCP ─────────────────────────────────────────────────────────────────────
LISTEN_HOST = '127.0.0.1'
LISTEN_PORT = 9876
POINT_BYTES = 12

# ─── UDP (RoboRIO) ────────────────────────────────────────────────────────────
ROBORIO_IP      = '10.0.0.2'
ROBORIO_PORT    = 5800
SEND_TO_ROBORIO = False

# ─── SCENE ───────────────────────────────────────────────────────────────────
X_RANGE = (-5.0, 5.0)
Z_RANGE = (-5.0, 5.0)

# ─── FILTERS ─────────────────────────────────────────────────────────────────
Y_OBS_MIN = -0.5   # allow points below sensor plane (back detection)
Y_OBS_MAX =  2.0   # ceiling cutoff
MIN_DIST  =  0.3
MAX_DIST  =  6.0
OBS_MEMORY = 5.0   # seconds to keep points

# ─── OBSTACLE PROCESSING ─────────────────────────────────────────────────────
VOXEL_SIZE  = 0.25
MIN_CLUSTER = 2

# ─── APF ─────────────────────────────────────────────────────────────────────
NX    = 60
NZ    = 60
K_ATT = 4.0
K_REP = 1500.0
D0_M  = 1.2

PATH_STEP     = 0.15
PATH_MAX_ITER = 3000

# ─── DRAW RATE ───────────────────────────────────────────────────────────────
DRAW_INTERVAL = 0.5   # seconds between plot updates (2 fps) — smooth and readable

# ─── NAVIGATION ──────────────────────────────────────────────────────────────
GOAL_M    = np.array([3.0, 0.0])
GOAL_LOCK = threading.Lock()

# ─── SHARED STATE (written by compute thread, read by draw thread) ────────────
render_lock  = threading.Lock()
render_state = {
    'obstacles' : [],
    'path'      : np.zeros((2, 2)),
    'U'         : None,
    'goal'      : np.array([3.0, 0.0]),
    'frame'     : 0,
    'n_recv'    : 0,
    'n_obs'     : 0,
}

# ─── BUFFERS ─────────────────────────────────────────────────────────────────
obs_buffer = collections.deque(maxlen=40000)
data_lock  = threading.Lock()

xs = np.linspace(*X_RANGE, NX)
zs = np.linspace(*Z_RANGE, NZ)
GX, GZ = np.meshgrid(xs, zs, indexing='ij')

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ─── ROBORIO ─────────────────────────────────────────────────────────────────
def send_to_roborio(path):
    if not SEND_TO_ROBORIO or len(path) < 2:
        return
    idx  = np.linspace(0, len(path)-1, min(20, len(path)), dtype=int)
    wpts = path[idx]
    n    = len(wpts)
    data = struct.pack(f'<H{n*2}f', n,
                       *[v for wp in wpts for v in (float(wp[0]), float(wp[1]))])
    try:
        udp_sock.sendto(data, (ROBORIO_IP, ROBORIO_PORT))
    except Exception as e:
        print(f"[RIO] {e}")

# ─── THREAD 1: TCP RECEIVER ──────────────────────────────────────────────────
def socket_receiver():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((LISTEN_HOST, LISTEN_PORT))
    except Exception as e:
        print(f"[TCP] BIND FAILED: {e}")
        return
    server.listen(1)
    print(f"[TCP] Waiting for lidar_bridge on port {LISTEN_PORT}…")

    while True:
        conn, addr = server.accept()
        print(f"[TCP] Bridge connected from {addr}")
        buf    = b''
        t_last = time.time()
        n_recv = n_obs = 0

        try:
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    print(f"[TCP] Disconnected  recv={n_recv} obs={n_obs}")
                    break
                buf += chunk
                now  = time.time()
                new_pts = []

                while len(buf) >= POINT_BYTES:
                    x, y, z = struct.unpack_from('<fff', buf)
                    buf = buf[POINT_BYTES:]
                    n_recv += 1

                    dist = np.sqrt(x*x + y*y + z*z)
                    if (MIN_DIST <= dist <= MAX_DIST and
                        Y_OBS_MIN <= y <= Y_OBS_MAX and
                        X_RANGE[0] <= x <= X_RANGE[1] and
                        Z_RANGE[0] <= z <= Z_RANGE[1]):
                        new_pts.append((x, z, now))
                        n_obs += 1

                if new_pts:
                    with data_lock:
                        obs_buffer.extend(new_pts)
                    with render_lock:
                        render_state['n_recv'] = n_recv
                        render_state['n_obs']  = n_obs

                if now - t_last > 5.0:
                    pct = 100 * n_obs / max(n_recv, 1)
                    print(f"[TCP] recv={n_recv} obs={n_obs} ({pct:.1f}%) "
                          f"buf={len(obs_buffer)}")
                    if pct < 5:
                        print(f"[TCP] WARNING: very few obstacle points — "
                              f"check Y_OBS_MIN={Y_OBS_MIN} Y_OBS_MAX={Y_OBS_MAX}")
                    t_last = now

        except Exception as e:
            print(f"[TCP] Error: {e}")
        finally:
            conn.close()

# ─── THREAD 2: COMPUTE WORKER ────────────────────────────────────────────────
def compute_worker():
    """
    Runs APF + path planning continuously in background.
    Writes results to render_state for the draw thread to read.
    Never touches matplotlib — no GUI calls here.
    """
    print("[COMPUTE] Worker started")
    frame = 0

    while True:
        t_start = time.time()
        now = time.time()

        # 1. Snapshot goal
        with GOAL_LOCK:
            goal = GOAL_M.copy()

        # 2. Get fresh obstacles
        with data_lock:
            fresh = [(x, z) for x, z, ts in obs_buffer
                     if now - ts < OBS_MEMORY]

        # 3. Voxelise + cluster filter
        obstacles = []
        if fresh:
            arr    = np.array(fresh, dtype=np.float32)
            idx    = np.floor(arr / VOXEL_SIZE).astype(np.int32)
            unique = np.unique(idx, axis=0)
            vox_set = set(map(tuple, unique))
            kept = []
            for vox in unique:
                n = sum(1 for dx in (-1,0,1) for dz in (-1,0,1)
                        if not (dx==0 and dz==0)
                        and (vox[0]+dx, vox[1]+dz) in vox_set)
                if n >= MIN_CLUSTER - 1:
                    kept.append(vox)
            if kept:
                kv = np.array(kept, dtype=np.float32)
                obstacles = ((kv + 0.5) * VOXEL_SIZE).tolist()

        # 4. APF
        U = 0.5 * K_ATT * ((GX - goal[0])**2 + (GZ - goal[1])**2)
        for ox, oz in obstacles:
            D = np.sqrt((GX-ox)**2 + (GZ-oz)**2)
            D[D < 0.05] = 0.05
            mask = D < D0_M
            U[mask] += 0.5 * K_REP * (1.0/D[mask] - 1.0/D0_M)**2

        # 5. Gradient descent path planning with local minimum escape
        #    Runs until goal is reached — no iteration cap
        gx_g = np.gradient(U, xs, axis=0)
        gz_g = np.gradient(U, zs, axis=1)

        def interp(G, px, pz):
            ix = int(np.clip(np.searchsorted(xs, px)-1, 0, NX-2))
            iz = int(np.clip(np.searchsorted(zs, pz)-1, 0, NZ-2))
            tx = np.clip((px-xs[ix])/(xs[ix+1]-xs[ix]+1e-9), 0, 1)
            tz = np.clip((pz-zs[iz])/(zs[iz+1]-zs[iz]+1e-9), 0, 1)
            return (G[ix,iz]    *(1-tx)*(1-tz) +
                    G[ix+1,iz]  *   tx *(1-tz) +
                    G[ix,iz+1]  *(1-tx)*   tz  +
                    G[ix+1,iz+1]*   tx *   tz)

        pos        = np.zeros(2)
        path       = [pos.copy()]
        stuck_count = 0
        prev_pos   = pos.copy()
        MAX_STEPS  = 5000    # safety cap — restarts if truly stuck
        step       = 0

        while step < MAX_STEPS:
            step += 1
            pos = np.clip(pos, [X_RANGE[0], Z_RANGE[0]],
                               [X_RANGE[1], Z_RANGE[1]])

            gx = interp(gx_g, pos[0], pos[1])
            gz = interp(gz_g, pos[0], pos[1])
            n  = np.sqrt(gx*gx + gz*gz)

            if n < 1e-4:
                # Local minimum — add random perturbation to escape
                stuck_count += 1
                perturb = np.random.uniform(-0.8, 0.8, 2)
                pos = pos + perturb
                print(f"[PATH] Local min escape #{stuck_count} "
                      f"at ({pos[0]:.2f},{pos[1]:.2f})")
                continue

            # Gradient descent step
            pos = pos - PATH_STEP * np.array([gx, gz]) / n
            path.append(pos.copy())

            # Check if we moved at all (oscillating)
            if step % 50 == 0:
                moved = np.linalg.norm(pos - prev_pos)
                if moved < PATH_STEP * 0.5:
                    # Oscillating around a minimum — bigger escape kick
                    stuck_count += 1
                    perturb = np.random.uniform(-1.2, 1.2, 2)
                    pos = pos + perturb
                    print(f"[PATH] Oscillation escape #{stuck_count}")
                prev_pos = pos.copy()

            # Goal reached
            if np.linalg.norm(pos - goal) < PATH_STEP * 2:
                path.append(goal.copy())
                print(f"[PATH] Goal reached in {step} steps "
                      f"escapes={stuck_count}")
                break

        path_arr = np.array(path)
        if step >= MAX_STEPS:
            print(f"[PATH] Safety cap hit — replanning next frame "
                  f"dist={np.linalg.norm(pos-goal):.2f}m")
        frame += 1

        # 6. Push results to render state
        with render_lock:
            render_state['obstacles'] = obstacles
            render_state['path']      = path_arr
            render_state['U']         = np.clip(U, 0, 600)
            render_state['goal']      = goal.copy()
            render_state['frame']     = frame

        send_to_roborio(path_arr)

        elapsed = time.time() - t_start
        print(f"[COMPUTE] frame={frame} obs={len(obstacles)} "
              f"path={len(path_arr)}pts compute={elapsed*1000:.0f}ms")

        # Small sleep to avoid hammering CPU
        time.sleep(0.1)

# ─── MAIN THREAD: DRAW LOOP ──────────────────────────────────────────────────
if __name__ == '__main__':

    print("[MAIN] Starting threads…")
    threading.Thread(target=socket_receiver, daemon=True).start()
    threading.Thread(target=compute_worker,  daemon=True).start()

    # ── Plot setup ───────────────────────────────────────────────────────────
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    apf_norm = Normalize(vmin=0, vmax=600)

    legend_handles = [
        mpatches.Patch(color='#ff4444', label='■ Obstacle'),
        mpatches.Patch(color='yellow',  label='── Path'),
        mpatches.Patch(color='#00ff88', label='★ Goal (click to move)'),
        mpatches.Patch(color='cyan',    label='● Rover'),
    ]

    def on_click(event):
        global GOAL_M
        if event.inaxes != ax or event.button != 1:
            return
        with GOAL_LOCK:
            GOAL_M = np.array([float(event.xdata), float(event.ydata)])
        print(f"[GOAL] → X={GOAL_M[0]:.2f}m  Z={GOAL_M[1]:.2f}m")

    fig.canvas.mpl_connect('button_press_event', on_click)
    print("[MAIN] Left-click map to set goal. Window will update smoothly.")

    t_last_draw = 0.0

    try:
        while plt.fignum_exists(fig.number):

            # Throttle redraws
            now = time.time()
            if now - t_last_draw < DRAW_INTERVAL:
                plt.pause(0.05)
                continue
            t_last_draw = now

            # Snapshot render state
            with render_lock:
                obstacles = list(render_state['obstacles'])
                path      = render_state['path'].copy()
                U_plot    = render_state['U']
                goal      = render_state['goal'].copy()
                frame     = render_state['frame']
                n_recv    = render_state['n_recv']
                n_obs     = render_state['n_obs']

            if U_plot is None:
                plt.pause(0.1)
                continue

            # ── Redraw ───────────────────────────────────────────────────────
            ax.cla()
            ax.set_facecolor('#161b22')

            # APF field
            ax.contourf(xs, zs, U_plot.T,
                        levels=np.linspace(0, 600, 50),
                        cmap='Blues_r', norm=apf_norm,
                        extend='neither', alpha=0.6)

            # Obstacles — red squares
            if obstacles:
                ox = [o[0] for o in obstacles]
                oz = [o[1] for o in obstacles]
                ax.scatter(ox, oz, c='#ff4444', s=300, marker='s',
                           edgecolors='#ff0000', linewidths=1.5, zorder=7)
                for x_o, z_o in obstacles:
                    d = np.sqrt(x_o**2 + z_o**2)
                    ax.annotate(f'{d:.1f}m', xy=(x_o, z_o),
                                xytext=(0, 8), textcoords='offset points',
                                color='white', fontsize=7,
                                ha='center', fontweight='bold')

            # Path
            if len(path) > 1:
                ax.plot(path[:,0], path[:,1], '-',
                        color='yellow', lw=2.5, zorder=6, alpha=0.9)
                wpts = path[::10]
                ax.scatter(wpts[:,0], wpts[:,1],
                           c='yellow', s=30, zorder=7, alpha=0.8)

            # Goal
            ax.plot(goal[0], goal[1], '*', color='#00ff88',
                    markersize=20, zorder=9,
                    markeredgecolor='white', markeredgewidth=0.5)
            ax.annotate('GOAL', xy=(goal[0], goal[1]),
                        xytext=(0, 14), textcoords='offset points',
                        color='#00ff88', fontsize=9,
                        ha='center', fontweight='bold')

            # Rover
            ax.plot(0, 0, 'o', color='cyan', markersize=14,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=9)
            ax.annotate('ROVER', xy=(0, 0),
                        xytext=(0, -16), textcoords='offset points',
                        color='cyan', fontsize=9,
                        ha='center', fontweight='bold')

            # Scan circle
            ax.add_patch(plt.Circle((0,0), MAX_DIST, fill=False,
                                    color='cyan', linestyle='--',
                                    lw=0.7, alpha=0.2))

            # Axes
            ax.set_xlim(*X_RANGE)
            ax.set_ylim(*Z_RANGE)
            ax.set_xlabel('X — forward / backward (m)',
                          color='white', fontsize=11)
            ax.set_ylabel('Z — left / right (m)',
                          color='white', fontsize=11)
            ax.tick_params(colors='white', labelsize=9)
            ax.grid(True, color='#30363d', linestyle='--',
                    linewidth=0.5, alpha=0.5)
            for sp in ax.spines.values():
                sp.set_edgecolor('#30363d')
            ax.legend(handles=legend_handles, loc='upper left',
                      fontsize=9, facecolor='#0d1117',
                      labelcolor='white', edgecolor='#30363d')

            pct = int(100 * n_obs / max(n_recv, 1))
            rio = ' → RoboRIO' if SEND_TO_ROBORIO else ''
            ax.set_title(
                f'Unitree L2 Rover{rio}  │  '
                f'obstacles: {len(obstacles)}  │  '
                f'path: {len(path)}pts  │  '
                f'pts: {n_recv} ({pct}% obs)',
                color='white', fontsize=10, pad=12
            )

            fig.canvas.draw_idle()
            plt.pause(0.05)

    except KeyboardInterrupt:
        print("\n[MAIN] Stopped.")
    finally:
        plt.ioff()
        plt.show()
