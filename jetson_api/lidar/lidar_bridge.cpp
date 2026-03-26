/**
 * lidar_bridge.cpp
 * ─────────────────────────────────────────────────────────────────────────────
 * Uses the official Unitree unilidar_sdk2 to receive point cloud data from
 * the L2 over serial, then forwards every valid XYZ point to the Python APF
 * over a local TCP socket as packed binary float32s.
 *
 * Wire format sent to Python (12 bytes per point):
 *   float32 x, float32 y, float32 z  (metres, Unitree right-hand frame)
 *
 * REVISED: Auto-reconnects to Python if connection drops (doesn't exit).
 *
 * BUILD:  see build.sh
 * RUN:    cd ~/unilidar_sdk2/unitree_lidar_sdk/bin && sudo ./lidar_bridge
 */

#include <iostream>
#include <cstring>
#include <cmath>
#include <thread>
#include <chrono>

// Correct include — example.h pulls in unitree_lidar_sdk.h + utilities
#include "example.h"

// POSIX sockets
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <atomic>
#include <mutex>

using namespace unilidar_sdk2;

// ── Config ────────────────────────────────────────────────────────────────────

static const char*    SERIAL_PORT  = "/dev/ttyACM0";
static const uint32_t BAUD         = 4000000;
static const char*    PYTHON_HOST  = "127.0.0.1";
static const int      PYTHON_PORT  = 9876;
static const float    MIN_DIST      = 0.0f;   // widened for debugging
static const float    MAX_DIST      = 100.0f; // widened for debugging
static const int      CMD_PORT      = 9877;   // FastAPI sends commands here
static const int      RECONNECT_RETRY_INTERVAL_S = 5;  // Wait 5s before retry

// ── Socket helpers ────────────────────────────────────────────────────────────

int connect_to_python() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return -1; }

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(PYTHON_PORT);
    inet_pton(AF_INET, PYTHON_HOST, &addr.sin_addr);

    for (int i = 0; i < 30; ++i) {
        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            std::cout << "[Bridge] Connected to Python on port "
                      << PYTHON_PORT << std::endl;
            return sock;
        }
        std::cout << "[Bridge] Waiting for Python APF to start ("
                  << i+1 << "/30)..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    close(sock);
    return -1;
}

bool send_point(int sock, float x, float y, float z) {
    float buf[3] = {x, y, z};
    const char* ptr = reinterpret_cast<const char*>(buf);
    int total = sizeof(buf), sent = 0;
    while (sent < total) {
        int n = send(sock, ptr + sent, total - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += n;
    }
    return true;
}

/**
 * Send a point with auto-reconnect logic.
 * If connection is lost, waits and reconnects instead of giving up.
 * Returns true if point was sent; false if should skip this point.
 */
bool send_point_with_reconnect(int& sock, float x, float y, float z) {
    // Try to send; if it fails, attempt to reconnect
    while (!send_point(sock, x, y, z)) {
        std::cerr << "[Bridge] Python disconnected, attempting reconnect..." << std::endl;
        close(sock);
        sock = -1;

        // Keep trying to reconnect
        while (sock < 0) {
            std::cout << "[Bridge] Reconnecting to Python in "
                      << RECONNECT_RETRY_INTERVAL_S << "s..." << std::endl;
            std::this_thread::sleep_for(
                std::chrono::seconds(RECONNECT_RETRY_INTERVAL_S));

            sock = connect_to_python();
            if (sock < 0) {
                std::cerr << "[Bridge] Reconnect failed, will retry..." << std::endl;
                continue;
            }
            std::cout << "[Bridge] Reconnected successfully" << std::endl;
        }

        // Now try to send again
        if (send_point(sock, x, y, z)) {
            return true;  // Success
        }
        // If it fails again, loop will retry connection
    }
    return true;
}

// ── Globals ──────────────────────────────────────────────────────────────────

UnitreeLidarReader* g_reader = nullptr;
std::mutex          g_reader_mutex;
std::atomic<bool>   g_restart_requested{false};

// ── Command socket thread ─────────────────────────────────────────────────────
// Listens on CMD_PORT for single-word commands from FastAPI.
// Supported: "RESTART"

void command_server_thread() {
    int server = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(CMD_PORT);

    if (bind(server, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "[CMD] Bind failed on port " << CMD_PORT << std::endl;
        return;
    }
    listen(server, 5);
    std::cout << "[CMD] Command server listening on port " << CMD_PORT << std::endl;

    while (true) {
        int conn = accept(server, nullptr, nullptr);
        if (conn < 0) continue;

        char buf[64] = {};
        int  n       = recv(conn, buf, sizeof(buf)-1, 0);
        if (n > 0) {
            std::string cmd(buf, n);
            // Trim whitespace
            while (!cmd.empty() && (cmd.back() == '\n' ||
                                    cmd.back() == '\r' ||
                                    cmd.back() == ' '))
                cmd.pop_back();

            std::cout << "[CMD] Received command: '" << cmd << "'" << std::endl;

            if (cmd == "RESTART") {
                std::lock_guard<std::mutex> lock(g_reader_mutex);
                if (g_reader) {
                    std::cout << "[CMD] Restarting LiDAR..." << std::endl;
                    g_reader->resetLidar();
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    g_reader->startLidarRotation();
                    std::cout << "[CMD] LiDAR restarted OK" << std::endl;
                    const char* resp = "OK: LiDAR restarted\n";
                    send(conn, resp, strlen(resp), 0);
                } else {
                    const char* resp = "ERROR: LiDAR not initialised\n";
                    send(conn, resp, strlen(resp), 0);
                }
            } else {
                std::string resp = "ERROR: Unknown command '" + cmd + "'\n";
                send(conn, resp.c_str(), resp.size(), 0);
                std::cout << "[CMD] Unknown command: " << cmd << std::endl;
            }
        }
        close(conn);
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "[Bridge] Starting LiDAR bridge (with auto-reconnect)..." << std::endl;

    // 1. Connect to Python first
    std::cout << "[Bridge] Connecting to Python APF..." << std::endl;
    int sock = connect_to_python();
    if (sock < 0) {
        std::cerr << "[Bridge] Could not connect initially. Start python3 lidar_apf.py first."
                  << std::endl;
        std::cerr << "[Bridge] Bridge will keep trying to connect..." << std::endl;
        sock = -1;  // Mark as disconnected, will auto-reconnect
    }

    // 2. Initialise SDK reader
    UnitreeLidarReader* lreader = createUnitreeLidarReader();

    std::cout << "[Bridge] Opening " << SERIAL_PORT
              << " @ " << BAUD << " baud..." << std::endl;

    // Explicitly pass all params for full 360 coverage:
    // cloud_scan_num=18  → all 18 vertical rings (full hemisphere)
    // range_min=0.0      → no SDK-level near cutoff (Python filters)
    // range_max=30.0     → L2 max rated range in metres
    if (lreader->initializeSerial(SERIAL_PORT, BAUD, 18, true, 0.0f, 30.0f) != 0) {
        std::cerr << "[Bridge] Serial init failed. "
                     "Check /dev/ttyACM0 permissions and UART mode." << std::endl;
        if (sock >= 0) close(sock);
        return 1;
    }
    std::cout << "[Bridge] LiDAR initialised OK." << std::endl;

    // Store in global so command thread can access it
    {
        std::lock_guard<std::mutex> lock(g_reader_mutex);
        g_reader = lreader;
    }

    // Start command server thread (for FastAPI restart commands)
    std::thread cmd_thread(command_server_thread);
    cmd_thread.detach();
    std::cout << "[Bridge] Command server started on port " << CMD_PORT << std::endl;

    // Follow exact SDK example sequence from Unitree docs:
    // startRotation → setWorkMode(8) → resetLidar → parse
    lreader->startLidarRotation();
    std::cout << "[Bridge] startLidarRotation() done, waiting 1s..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "[Bridge] setLidarWorkMode(8)..." << std::endl;
    lreader->setLidarWorkMode(8);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "[Bridge] resetLidar()..." << std::endl;
    lreader->resetLidar();
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::cout << "[Bridge] LiDAR ready — entering parse loop" << std::endl;

    // 3. Parse loop (runs forever now, reconnects on disconnect)
    uint64_t pts_sent   = 0;
    uint64_t frames     = 0;
    uint64_t reconnects = 0;
    auto     t_last     = std::chrono::steady_clock::now();

    std::cout << "[Bridge] Streaming points to Python..." << std::endl;

    while (true) {
        int ret = lreader->runParse();

        // LIDAR_POINT_DATA_PACKET_TYPE = 102
        if (ret == 102) {
            PointCloudUnitree cloud;
            if (lreader->getPointCloud(cloud)) {
                ++frames;
                // Debug: print first 3 points of each frame
                if (frames <= 3 && !cloud.points.empty()) {
                    std::cout << "[DEBUG] Frame " << frames
                              << " cloud size=" << cloud.points.size()
                              << " first pt: x=" << cloud.points[0].x
                              << " y=" << cloud.points[0].y
                              << " z=" << cloud.points[0].z
                              << " dist=" << std::sqrt(
                                  cloud.points[0].x*cloud.points[0].x +
                                  cloud.points[0].y*cloud.points[0].y +
                                  cloud.points[0].z*cloud.points[0].z)
                              << std::endl;
                }

                for (const auto& pt : cloud.points) {
                    float dist = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
                    if (dist < MIN_DIST || dist > MAX_DIST) continue;

                    // Auto-reconnect on disconnect
                    send_point_with_reconnect(sock, pt.x, pt.y, pt.z);
                    ++pts_sent;
                }
            }

            // Diagnostics every 5 s
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_last).count();
            if (elapsed >= 5.0) {
                std::cout << "[Bridge] frames=" << frames
                          << "  pts_sent=" << pts_sent
                          << "  pts/frame~" << pts_sent / std::max(frames, (uint64_t)1)
                          << "  reconnects=" << reconnects
                          << std::endl;
                t_last = now;
            }
        }
    }

    // Cleanup (rarely reached)
    lreader->stopLidarRotation();
    if (sock >= 0) close(sock);
    return 0;
}