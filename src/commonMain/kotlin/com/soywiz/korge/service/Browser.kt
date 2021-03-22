package com.soywiz.korge.service

import com.soywiz.korgw.*
import com.soywiz.korio.file.*

expect object BrowserImpl {
    suspend fun alert(msg: String)
    suspend fun prompt(message: String, defaultValue: String): String
    suspend fun openFileDialog(): List<VfsFile>

}

class Browser(val gameWindow: GameWindow) {
    suspend fun alert(msg: String) {
        //return gameWindow.alert(msg)
        return BrowserImpl.alert(msg)
    }

    suspend fun prompt(message: String, defaultValue: String): String {
        //return gameWindow.prompt(message, title)
        return BrowserImpl.prompt(message, defaultValue)
    }

    suspend fun openFile(): VfsFile? {
        //return gameWindow.openFileDialog().first()
        return BrowserImpl.openFileDialog().firstOrNull()
    }
}
