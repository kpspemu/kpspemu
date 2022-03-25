package com.soywiz.korge.service

import com.soywiz.klock.*
import com.soywiz.korio.file.*
import com.soywiz.korio.file.std.*
import kotlinx.coroutines.*
import java.awt.*
import javax.swing.*

actual object BrowserImpl {
    actual suspend fun alert(msg: String) {
        JOptionPane.showMessageDialog(null, msg, "Alert", JOptionPane.INFORMATION_MESSAGE);
    }

    actual suspend fun prompt(message: String, defaultValue: String): String {
        return JOptionPane.showInputDialog(null, message, defaultValue)
    }

    actual suspend fun openFileDialog(): List<VfsFile> {
        val deferred = CompletableDeferred<List<VfsFile>>()
        EventQueue.invokeLater {
            val dialog = FileDialog(null as Frame?)
            dialog.isVisible = true
            deferred.complete(dialog.files.map { localVfs(it.parentFile)[it.name] })
        }
        return deferred.await()
    }
}