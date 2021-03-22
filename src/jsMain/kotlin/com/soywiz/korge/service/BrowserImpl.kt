package com.soywiz.korge.service

import com.soywiz.korio.file.*

actual object BrowserImpl {
    actual suspend fun alert(msg: String) {
    }

    actual suspend fun prompt(message: String, defaultValue: String): String {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    actual suspend fun openFileDialog(): List<VfsFile> {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }
}