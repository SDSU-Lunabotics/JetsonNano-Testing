/**
 * lidar_bridge.cpp
 * ═══════════════════════════════════════════════════════════════
 * Reads point cloud from Unitree L2 via SDK over serial (UART),
 * forwards every point as raw XYZ float32 over TCP to Python.
 *
 * Wire format (12 bytes per point, little-endian):
 *   float32 x  (metres)
 *   float32 y  (metres)
 *   float32 z  (metres)
 *
 * BUILD:
 *   chmod +x build.sh && ./build.sh
 *
 * RUN (after python3 lidar_rover.py is already listening):
 *   cd ~/unilidar_sdk2/unitree_lidar_sdk/bin
 *   ./lidar_bridge
 * ═══════════════════════════════════════════════════════════════
 */

#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>
#include "example.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

using namespace unilidar_sdk2;

// ── Config ────────────────────────────────────────────────────────────────────

static const char*    SERIAL_PORT = "/dev/ttyACM0";
static const uint32_t BAUD        = 4000000;
static const char*    PYTHON_HOST = "127.0.0.1";
static const int      PYTHON_PORT = 9876;
static const int      CMD_PORT    = 9877;

// ── Globals ───────────────────────────────────────────────────────────────────

UnitreeLidarReader* g_reader = nullptr;
std::mutex          g_mutex;

// ── Helpers ───────────────────────────────────────────────────────────────────

int connect_to_python() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(PYTHON_PORT);
    inet_pton(AF_INET, PYTHON_HOST, &addr.sin_addr);

    for (int i = 0; i < 60; ++i) {
        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            std::cout << "[Bridge] Connected to Python on port "
                      << PYTHON_PORT << std::endl;
            return sock;
        }
        std::cout << "[Bridge] Waiting for Python... ("
                  << i+1 << "/60)" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    close(sock);
    return -1;
}

bool send_xyz(int sock, float x, float y, float z) {
    float buf[3] = {x, y, z};
    const char* ptr = reinterpret_cast<const char*>(buf);
    int sent = 0;
    while (sent < 12) {
        int n = send(sock, ptr + sent, 12 - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += n;
    }
    return true;
}

// ── Command server ────────────────────────────────────────────────────────────

void cmd_server() {
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
    listen(server, 3);
    std::cout << "[CMD] Command server on port " << CMD_PORT << std::endl;

    while (true) {
        int conn = accept(server, nullptr, nullptr);
        if (conn < 0) continue;
        char buf[64] = {};
        int n = recv(conn, buf, sizeof(buf)-1, 0);
        if (n > 0) {
            std::string cmd(buf, n);
            while (!cmd.empty() &&
                   (cmd.back()=='\n'||cmd.back()=='\r'||cmd.back()==' '))
                cmd.pop_back();
            std::cout << "[CMD] Received: '" << cmd << "'" << std::endl;
            if (cmd == "RESTART") {
                std::lock_guard<std::mutex> lk(g_mutex);
                if (g_reader) {
                    g_reader->startLidarRotation();
                    const char* r = "OK\n";
                    send(conn, r, strlen(r), 0);
                    std::cout << "[CMD] LiDAR restarted" << std::endl;
                }
            }
        }
        close(conn);
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
    // 1. Connect to Python first
    std::cout << "[Bridge] Looking for Python on "
              << PYTHON_HOST << ":" << PYTHON_PORT << std::endl;
    int sock = connect_to_python();
    if (sock < 0) {
        std::cerr << "[Bridge] Could not connect. "
                     "Start python3 lidar_rover.py first." << std::endl;
        return 1;
    }

    // 2. Initialise SDK
    UnitreeLidarReader* lreader = createUnitreeLidarReader();
    std::cout << "[Bridge] Opening " << SERIAL_PORT
              << " @ " << BAUD << " baud..." << std::endl;

    if (lreader->initializeSerial(SERIAL_PORT, BAUD, 18, true, 0.0f, 30.0f) != 0) {
        std::cerr << "[Bridge] Serial init failed. Check:" << std::endl;
        std::cerr << "  1. L2 is in UART mode (via Unilidar 2)" << std::endl;
        std::cerr << "  2. sudo chmod a+rw " << SERIAL_PORT << std::endl;
        std::cerr << "  3. Correct port name" << std::endl;
        close(sock);
        return 1;
    }
    std::cout << "[Bridge] Serial OK" << std::endl;

    // Store global ref for command server
    { std::lock_guard<std::mutex> lk(g_mutex); g_reader = lreader; }

    // 3. Start command server thread
    std::thread(cmd_server).detach();

    // 4. Startup — clear buffer and go straight to parsing
    // Do NOT call startLidarRotation() or setLidarWorkMode() —
    // both cause a serial restart → timeout warnings
    std::cout << "[Bridge] Clearing buffer..." << std::endl;
    lreader->clearBuffer();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "[Bridge] Ready. Streaming points..." << std::endl;

    // 5. Parse loop — runs forever
    uint64_t frames = 0;
    uint64_t pts    = 0;
    auto     t_diag = std::chrono::steady_clock::now();

    while (true) {
        std::cerr << "[DEBUG] main loop restarting " << std::endl;
        int ret = lreader->runParse();
        std::cerr << "[DEBUG] ret is " << ret << std::endl;

        // SDK revisions may report point-data packets as 102 or 104.
        if (ret == 102 || ret == 104) {
            PointCloudUnitree cloud;
            if (!lreader->getPointCloud(cloud)) continue;

            ++frames;

            // ── DEBUG: print first 5 frames ─────────────────────────────
            if (frames <= 5) {
                std::cout << "[DEBUG] Frame " << frames
                          << "  size=" << cloud.points.size();
                if (!cloud.points.empty()) {
                    const auto& p0 = cloud.points[0];
                    float d = std::sqrt(p0.x*p0.x + p0.y*p0.y + p0.z*p0.z);
                    std::cout << "  pt[0]: x=" << p0.x
                              << " y=" << p0.y
                              << " z=" << p0.z
                              << " dist=" << d << "m";

                    // Print Y range of whole frame
                    float y_min = p0.y, y_max = p0.y;
                    for (const auto& pt : cloud.points) {
                        y_min = std::min(y_min, pt.y);
                        y_max = std::max(y_max, pt.y);
                    }
                    std::cout << "  Y_range=[" << y_min
                              << ", " << y_max << "]";
                }
                std::cout << std::endl;
            }
            std::cerr << "[DEBUG] begin processing points " << std::endl;

            // ── Stream all points to Python ──────────────────────────────
            for (const auto& pt : cloud.points) {
                if (!send_xyz(sock, pt.x, pt.y, pt.z)) {
                    std::cout << "[Bridge] Python disconnected. "
                                 "Reconnecting..." << std::endl;
                    close(sock);
                    sock = connect_to_python();
                    if (sock < 0) {
                        std::cerr << "[Bridge] Could not reconnect."
                                  << std::endl;
                        lreader->stopLidarRotation();
                        return 1;
                    }
                    break;
                }
                ++pts;
            }
            std::cerr << "[DEBUG] end processing points " << std::endl;

        }

        // ── Diagnostics every 5 s ────────────────────────────────────────
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t_diag).count();
        if (elapsed >= 5.0) {
            std::cout << "[Bridge] frames=" << frames
                      << "  pts=" << pts
                      << "  pts/frame=" << pts / std::max(frames, (uint64_t)1)
                      << std::endl;
            t_diag = now;
        }
    }

    lreader->stopLidarRotation();
    close(sock);
    return 0;
}