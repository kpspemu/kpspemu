package com.soywiz.korge.service

import com.soywiz.korgw.*
import com.soywiz.korio.file.*

class Browser(val gameWindow: GameWindow) {
    suspend fun alert(msg: String) {
        return gameWindow.alert(msg)
    }

    suspend fun prompt(message: String, title: String): String {
        return gameWindow.prompt(message, title)
    }

    suspend fun openFile(): VfsFile {
        return gameWindow.openFileDialog().first()
    }
}
