import threading
import time
from networktables import NetworkTables

from app.core.settings import settings


class NTService:
    def __init__(self) -> None:
        self._initialized = False
        self._table = None

    def initialize(self) -> None:
        if self._initialized:
            return
        NetworkTables.initialize(server=settings.roborio_ip)
        self._table = NetworkTables.getTable("SmartDashboard")
        self._initialized = True

    def is_connected(self) -> bool:
        self.initialize()
        return NetworkTables.isConnected()

    def wait_for_connection(self, timeout: float = 5.0) -> bool:
        self.initialize()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if NetworkTables.isConnected():
                return True
            time.sleep(0.1)
        return NetworkTables.isConnected()

    def set_led(self, state: bool, override: bool = True) -> None:
        self.initialize()
        self._table.putBoolean("Jetson/LedOverride", bool(override))
        self._table.putBoolean("Jetson/LedState", bool(state))

    def drive_forward(self, duration: float, speed: float) -> None:
        self.initialize()

        # NOTE:
        # This key name is kept for RoboRIO compatibility.
        # It does NOT mean true autonomy mode in the Jetson app state.
        self._table.putBoolean("Jetson/AutomationEnabled", True)

        self._table.putString("Jetson/Command", "drive_forward")
        self._table.putNumber("Jetson/CommandForward", float(speed))
        self._table.putNumber("Jetson/CommandTurn", 0.0)
        self._table.putNumber("Jetson/CommandDuration", float(duration))
        self._table.putBoolean("Jetson/CommandReady", True)

        time.sleep(0.1)
        self._table.putBoolean("Jetson/CommandReady", False)

        time.sleep(duration)
        self._table.putNumber("Jetson/CommandForward", 0.0)
        self._table.putNumber("Jetson/CommandTurn", 0.0)
        self._table.putNumber("Jetson/CommandDuration", 0.0)
        self._table.putBoolean("Jetson/AutomationEnabled", False)
        self._table.putString("Jetson/Command", "")

    def set_excavator(self, enabled: bool) -> None:
        self.initialize()
        self._table.putBoolean("Jetson/ExcavatorEnabled", bool(enabled))

    def set_excavator_lowering_sim(self, enabled: bool) -> None:
        self.initialize()
        self._table.putBoolean("Jetson/ExcavatorLoweringSim", bool(enabled))

    def set_excavator_left_extend(self, enabled: bool) -> None:
        self.initialize()
        self._table.putBoolean("Jetson/ExcavatorLeftExtend", bool(enabled))

    def set_excavator_right_extend(self, enabled: bool) -> None:
        self.initialize()
        self._table.putBoolean("Jetson/ExcavatorRightExtend", bool(enabled))

    def set_door_open(self, enabled: bool) -> None:
        self.initialize()
        self._table.putBoolean("Jetson/DoorActuatorsClose", False if enabled else self._table.getBoolean("Jetson/DoorActuatorsClose", False))
        self._table.putBoolean("Jetson/DoorActuatorsOpen", bool(enabled))

    def set_door_close(self, enabled: bool) -> None:
        self.initialize()
        self._table.putBoolean("Jetson/DoorActuatorsOpen", False if enabled else self._table.getBoolean("Jetson/DoorActuatorsOpen", False))
        self._table.putBoolean("Jetson/DoorActuatorsClose", bool(enabled))

    def stop_door(self) -> None:
        self.initialize()
        self._table.putBoolean("Jetson/DoorActuatorsOpen", False)
        self._table.putBoolean("Jetson/DoorActuatorsClose", False)

    def set_deposition(self, enabled: bool) -> None:
        self.initialize()
        self._table.putBoolean("Jetson/ConveyorEnabled", bool(enabled))

    def set_conveyor(self, enabled: bool) -> None:
        self.set_deposition(enabled)

    def stop_all_motion(self) -> None:
        self.initialize()
        self._table.putBoolean("Jetson/AutomationEnabled", False)
        self._table.putNumber("Jetson/CommandForward", 0.0)
        self._table.putNumber("Jetson/CommandTurn", 0.0)
        self._table.putNumber("Jetson/CommandDuration", 0.0)
        self._table.putBoolean("Jetson/CommandReady", False)
        self._table.putString("Jetson/Command", "")

        self._table.putBoolean("Jetson/ExcavatorEnabled", False)
        self._table.putBoolean("Jetson/ExcavatorLoweringSim", False)
        self._table.putBoolean("Jetson/ExcavatorLeftExtend", False)
        self._table.putBoolean("Jetson/ExcavatorRightExtend", False)
        self._table.putBoolean("Jetson/DoorActuatorsOpen", False)
        self._table.putBoolean("Jetson/DoorActuatorsClose", False)
        self._table.putBoolean("Jetson/ConveyorEnabled", False)

    def run_async(self, fn, *args) -> None:
        threading.Thread(target=fn, args=args, daemon=True).start()


nt_service = NTService()
