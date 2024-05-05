//
//  SDKManagerActor.swift
//  DjiMobileSdkTest
//
//  Created by Fenton Chance on 03.23.24
//

import DJISDK
import Foundation
import os.log

class SDKManagerActor: ObservableObject {
    
    private let logger = Logger(subsystem: "bundleId", category: "classname")
    
    func registerApp(with delegate: SDKManagerObserver) {
        let appKey = Bundle.main.object(forInfoDictionaryKey: SDK_APP_KEY_INFO_PLIST_KEY) as? String
        
        guard appKey != nil && appKey!.isEmpty == false else {
            logger.critical("Please enter your app key in the info.plist")
            return
        }
        DJISDKManager.registerApp(with: delegate)
    }
    
    func stop() {
        DJISDKManager.stopConnectionToProduct()
    }
}
