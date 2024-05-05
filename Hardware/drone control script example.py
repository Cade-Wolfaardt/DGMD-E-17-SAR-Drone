import socket
import threading
from pymavlink import mavutil

# Configuration
MAVLINK_CONNECTION_STRING = "/dev/serial0"  # Update to match your setup
BAUD_RATE = 57600

def initialize_mavlink_connection():
    """Initialize and return a MAVLink connection object."""
    try:
        return mavutil.mavlink_connection(MAVLINK_CONNECTION_STRING, baud=BAUD_RATE)
    except Exception as e:
        print(f"Failed to initialize MAVLink connection: {e}")
        return None

def handle_command(mavconn, command):
    """Parse and execute commands received from the TCP server."""
    if not mavconn:
        print("MAVLink connection is not initialized.")
        return

    args = command.split()
    cmd = args[0].lower()
    params = args[1:]

    try:
        if cmd == "takeoff":
            altitude = float(params[0]) if params else 10  # Default altitude
            mavconn.mav.command_long_send(
                mavconn.target_system, mavconn.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, altitude)
            print(f"Takeoff to {altitude} meters.")

        elif cmd == "land":
            mavconn.mav.command_long_send(
                mavconn.target_system, mavconn.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            print("Landing initiated.")

        elif cmd == "goto":
            lat, lon, alt = map(float, params)
            mavconn.mav.mission_item_send(
                mavconn.target_system, mavconn.target_component,
                0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0,
                0, 0, 0, 0, lat, lon, alt)
            print(f"Going to lat: {lat}, lon: {lon}, alt: {alt} meters.")

        elif cmd == "setalt":
            alt = float(params[0])
            mavconn.mav.command_long_send(
                mavconn.target_system, mavconn.target_component,
                mavutil.mavlink.MAV_CMD_CONDITION_CHANGE_ALT, 0, alt, 0, 0, 0, 0, 0, 0)
            print(f"Altitude change to {alt} meters requested.")

        elif cmd == "loiter":
            seconds = float(params[0]) if params else 60  # Default loiter time
            mavconn.mav.command_long_send(
                mavconn.target_system, mavconn.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LOITER_TIME, 0, seconds, 0, 0, 0, 0, 0, 0)
            print(f"Loitering for {seconds} seconds.")

        elif cmd == "returntohome":
            mavconn.mav.command_long_send(
                mavconn.target_system, mavconn.target_component,
                mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            print("Return to home command sent.")

        # Implement further elif blocks for additional commands as needed

    except Exception as e:
        print(f"Error processing command {cmd}: {e}")

def command_server(mavconn):
    host = 'localhost'
    port = 6001
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Command server listening on {host}:{port}")

    while True:
        client_socket, addr = server_socket.accept()
        with client_socket:
            while True:
                command = client_socket.recv(1024).decode()
                if not command:
                    break
                handle_command(mavconn, command)

def start_video_stream_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 5000))
    server_socket.listen()
    print("Video stream server listening on port 5000.")

    while True:
        client_socket, addr = server_socket.accept()
        with client_socket:
            while True:
                video_data = client_socket.recv(1024)
                if not video_data:
                    break
                # Process or forward the received video data

if __name__ == "__main__":
    mavlink_connection = initialize_mavlink_connection()
    threading.Thread(target=command_server, args=(mavlink_connection,)).start()
    threading.Thread(target=start_video_stream_server).start()

