import argparse
import time
from pymavlink import mavutil
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG, filename='sar_drone.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


# Connect to the Vehicle
def connect_vehicle(connection_string):
    if not connection_string:
        connection_string = 'udp:127.0.0.1:14551'  # default link port

    print('Connecting to vehicle on: %s' % connection_string)

    # Connect to SITL output using the appropriate IP and port
    vehicle = mavutil.mavlink_connection(connection_string, source_system=255, source_component=0)
    vehicle.wait_heartbeat()
    print(f"Heartbeat from system {vehicle.target_system} component {vehicle.target_component}")

    return vehicle


# Clear any existing mission
def clear_mission(vehicle) -> None:
    vehicle.mav.mission_clear_all_send(vehicle.target_system, vehicle.target_component)
    while True:
        msg = vehicle.recv_match(type='MISSION_ACK', blocking=True)
        if msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
            print("Mission cleared")
            break


# Read in mission file
def read_mission_file(filename: str) -> list:
    with open(filename, 'r') as file:
        first_line = file.readline()
        if not first_line.startswith('QGC WPL 110'):
            logging.debug(f"First Line Wrong Format: {first_line}")
            raise Exception('File format not supported')

        mission_items = []
        for line in file:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            mission_items.append(line)

    return mission_items


# Upload mission MAVLink command list to drone
def upload_mission(vehicle, mission):
    mission_items = []
    logging.info(f"Uploading Mission with {len(mission) -1} commands:")
    
    # confirm the mission has the right header
    assert mission[0] == "QGC WPL 110" 

    # convert mission into MAVLink commands
    for _, line in enumerate(mission):
        # skips the first line
        if line.startswith("QGC"):
            continue

        parts = line.split('\t')
        # logging.debug(parts)

        # the first command MUST be takeoff
        if int(parts[0]) == 0:
            assert int(parts[3]) == 22
        # the last command MUST be landing
        elif parts[0] == len(mission) - 2:
            assert int(parts[3]) == 20

        # add each command to mission command list
        lat, lon, alt = map(float, parts[8:11])
        seq = int(parts[0])
        frame = int(parts[2])
        command = int(parts[3])
        mission_items.append(mavutil.mavlink.MAVLink_mission_item_message(
                vehicle.target_system, vehicle.target_component, seq, frame,
                command, 0, 0, 0, 0, 0, 0, lat, lon, alt))


    # send over mission command list count first
    vehicle.mav.mission_count_send(vehicle.target_system, vehicle.target_component, len(mission_items))

    # send over mission commands
    for i, item in enumerate(mission_items):
        msg = vehicle.recv_match(type='MISSION_REQUEST', blocking=True)
        vehicle.mav.send(item)

    # See if mission upload complete
    msg = vehicle.recv_match(type='MISSION_ACK', blocking=True)

    if msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
        print("Mission upload SUCCESS!")
    else:
        print("Mission upload FAILED!")

    return len(mission_items)


# Arm and take off the drone
def arm_and_takeoff(vehicle, target_altitude):
    # Arm vehicle before attempting to take off
    print("Arming motors")
    vehicle.mav.command_long_send(
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0)  # 1 to arm

    # Wait for arming
    vehicle.motors_armed_wait()
    print("Motors armed!")

    print("Taking off!")
    vehicle.mav.command_long_send(
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, target_altitude)

    # Wait until the vehicle reaches a safe height
    while True:
        # Request altitude
        vehicle.mav.request_data_stream_send(vehicle.target_system, vehicle.target_component, mavutil.mavlink.MAV_DATA_STREAM_POSITION, 1, 1)
        msg = vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        altitude = msg.alt / 1000.0  # Convert from mm to m

        print("Altitude:", altitude)
        if altitude > target_altitude * 0.95:  # Just below target, in meters
            print("Reached target altitude")
            break
        time.sleep(1)


# Land the vehicle
def land_vehicle(vehicle):
    print("Landing")
    vehicle.mav.command_long_send(
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0)

    print("Landing command sent")


# Change vehicle mode
def set_mode(vehicle, target_mode):
    mode_id = vehicle.mode_mapping()[target_mode]

    # Send mode change command to AUTO mode
    vehicle.mav.command_long_send(
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id, 0, 0, 0, 0, 0)

    if command_ack(vehicle, mavutil.mavlink.MAV_CMD_DO_SET_MODE):
        print(f"Mode set to {target_mode}")
    else:
        print(f"FAILED to set mode to {target_mode}")


# Check GPS fix before changing mode
def check_gps_fix(vehicle):
    ack_msg = vehicle.recv_match(type='GPS_RAW_INT', blocking=True)
    if ack_msg.fix_type >= 3:  # Typically, a fix_type of 3 means a 3D fix
        print("GPS Fix OK.")
        return True
    else:
        print("Waiting for GPS Fix...")
        return False


# Check status message
def command_ack(vehicle, cmd_id):
    while True:
        ack_msg = vehicle.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
        if ack_msg is None:
            print("Waiting for command acknowledgement...")
            time.sleep(1)
            continue
        elif ack_msg.command == cmd_id:
            if ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                return True
            else:
                return False


# Launch Mission Execution Sequence
def run_mission(connection_string: str, mission_cmds) -> None:
    # 1. Mission Preparation
    copter = connect_vehicle(connection_string)
    clear_mission(copter)
#
    # 2. Upload mission
#    mission_cmds = read_mission_file(mission_file)
    waypoints = upload_mission(copter, mission_cmds)

    # 3. Pre-takeoff check
    while not check_gps_fix(copter):
        time.sleep(2)

    # 4. Switch to GUIDED mode
    set_mode(copter, "GUIDED")

    # 5. Arm and Take-off
    arm_and_takeoff(copter, 10)  # initial height 10m
    time.sleep(5)  # let the copter catches its breath, before ..

    # 6. Switch to Autonomous flight
    set_mode(copter, "AUTO")  # execute the mission

    # 7. Monitor mission progress
    current_wp = 0
    while True:
        next_wp = copter.recv_match(type='MISSION_CURRENT', blocking=True, timeout=5).seq
        if next_wp == current_wp:
            continue

        # We've reached a new waypoint
        print(f"Reached waypoint {next_wp}")
        current_wp = next_wp

        # Have we done all waypoints?
        if next_wp == waypoints - 1:
            # Land and finish
            land_vehicle(copter)

            # How 
            print("Mission Completed")
            break
        time.sleep(5)
    
    # 8. Mission Accomplished
    copter.close()


# Protected main
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='commands')
    parser.add_argument('--connect')
    parser.add_argument('--mission')
    parser.add_argument('--model')
    args = parser.parse_args()

    # Connect to quadcopter
    connection_string = args.connect
    mission_file = args.mission
    ppmodel = args.model

    print(f"Connection String: {connection_string}")
    print(f"Mission Data: {mission_file}")
    print(f"Path Planning Type: {ppmodel}")

    run_mission(connection_string, read_mission_file(mission_file))
