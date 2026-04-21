"""
Quick test script to verify Schunk EGK40 gripper connection and basic open/close.
Run with: python -m raspberry_IL.uR3station.test_schunk_gripper
"""

from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripperSpecs
from raspberry_IL.hardware.grippers.schunk_process import SchunkGripperProcess, SCHUNK_DEFAULT_SPECS

USB_PORT = "/dev/ttyUSB0"
FINGERTIP_GAP_MM = 41.0  # measured gap between fingertip faces when fully open

custom_specs = ParallelPositionGripperSpecs(
    max_width=FINGERTIP_GAP_MM / 1000,
    min_width=SCHUNK_DEFAULT_SPECS.min_width,
    max_force=SCHUNK_DEFAULT_SPECS.max_force,
    min_force=SCHUNK_DEFAULT_SPECS.min_force,
    max_speed=SCHUNK_DEFAULT_SPECS.max_speed,
    min_speed=SCHUNK_DEFAULT_SPECS.min_speed,
)

print(f"Connecting to Schunk EGK40 on {USB_PORT}...")
gripper = SchunkGripperProcess(USB_PORT, gripper_specs=custom_specs)
print("Connected.")
print(f"  max_width : {gripper.gripper_specs.max_width * 1000:.1f} mm")
print(f"  min_width : {gripper.gripper_specs.min_width * 1000:.1f} mm")
print(f"  current width: {gripper.get_current_width() * 1000:.1f} mm")

input("\nPress Enter to OPEN gripper...")
gripper.open()
print(f"  width after open: {gripper.get_current_width() * 1000:.1f} mm")

input("Press Enter to CLOSE gripper (to 20 mm)...")
gripper.move(0.020)
print(f"  width after close: {gripper.get_current_width() * 1000:.1f} mm")

input("Press Enter to OPEN again...")
gripper.open()
print(f"  width after open: {gripper.get_current_width() * 1000:.1f} mm")

print("\nDone.")
gripper.shutdown()
