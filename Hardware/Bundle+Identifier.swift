//
//  Bundle+Identifier.swift
//  DjiMobileSdkTest
//
//  Fenton Chance - 04.02.24
//

import Foundation

extension Bundle {
    static var theId: String {
        Bundle.main.bundleIdentifier ?? ""
    }
}
