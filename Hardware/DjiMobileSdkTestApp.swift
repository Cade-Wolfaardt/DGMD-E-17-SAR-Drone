//
//  DjiMobileSdkTestApp.swift
//  DjiMobileSdkTest
//
//  Fenton Chance - 03.11.24
//

import Combine
import DJISDK
import SwiftUI
import os.log

@main
struct DjiMobileSdkTestApp: App {
    private let logger = Logger(subsystem: Bundle.theId, category: "DjiMobileSdkTestApp")
    
    @Environment(\.scenePhase) private var scenePhase
    
    var sdkManagerProxy: SdkManagerProxy
    var baseProductProxy: BaseProductProxy
    var flightControllerProxy: FlightControllerProxy
    var remoteControllerProxy: RemoteControllerProxy
    
    var cancellables = [AnyCancellable]()
    
    init() {
        sdkManagerProxy = SdkManagerProxy()
        baseProductProxy = BaseProductProxy()
        flightControllerProxy = FlightControllerProxy()
        remoteControllerProxy = RemoteControllerProxy()
        
        sdkManagerProxy
            .$baseProduct
            .subscribe(baseProductProxy.subject)
            .store(in: &cancellables)
        
        sdkManagerProxy
            .$remoteController
            .subscribe(remoteControllerProxy.subject)
            .store(in: &cancellables)
        
        sdkManagerProxy
            .$flightController
            .subscribe(flightControllerProxy.subject)
            .store(in: &cancellables)
        
        DJISDKManager.closeConnection(whenEnteringBackground: false)
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
                .padding()
        }
        .onChange(of: scenePhase) { (newScenePhase) in
            switch newScenePhase {
            case .active:
                logger.debug("Scene entered active phase")
                sdkManagerProxy.registerApp()
            case .inactive:
                logger.debug("Scene entered inactive phase")
                DJISDKManager.stopConnectionToProduct()
            case .background:
                logger.debug("Scene entered background phase")
                DJISDKManager.stopConnectionToProduct()
            @unknown default:
                logger.debug("Scene entered an undetermined phase")
            }
        }
    }
}

