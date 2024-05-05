//
//  TCPClient.swift
//  DjiMobileSdkTest
//
//  Fenton Chance - 03.23.24
//

import Foundation
import Network

class DroneCommandClient {
    private var connection: NWConnection?
    private let host: NWEndpoint.Host
    private let port: NWEndpoint.Port

    init(host: String, port: UInt16) {
        self.host = NWEndpoint.Host(host)
        self.port = NWEndpoint.Port(rawValue: port) ?? NWEndpoint.Port(6001)
    }

    func connect() {
        connection = NWConnection(host: host, port: port, using: .tcp)
        connection?.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                print("Connected to the command server")
            case .failed(let error):
                print("Failed to connect: \(error)")
                self?.connection?.cancel()
            default:
                break
            }
        }
        connection?.start(queue: .main)
    }

    func sendCommand(_ command: String) {
        guard let connection = connection else {
            print("Connection not initialized")
            return
        }

        let commandWithNewline = command + "\n"
        if let data = commandWithNewline.data(using: .utf8) {
            connection.send(content: data, completion: .contentProcessed({ error in
                if let error = error {
                    print("Failed to send command: \(error)")
                } else {
                    print("Command sent: \(command)")
                }
            }))
        }
    }

    func disconnect() {
        connection?.cancel()
        connection = nil
    }
}

// Usage example:
let droneClient = DroneCommandClient(host: "localhost", port: 6001)
droneClient.connect()

// Sending commands:
droneClient.sendCommand("takeoff 20")
droneClient.sendCommand("land")
droneClient.sendCommand("goto 37.7749 -122.4194 100")
droneClient.sendCommand("returntohome")

// Disconnect when done:
droneClient.disconnect()

