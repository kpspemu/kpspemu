package com.soywiz.kpspemu.ui

import com.soywiz.korge.service.*
import com.soywiz.korge.view.*
import com.soywiz.korio.error.*
import com.soywiz.korio.file.std.*
import com.soywiz.korio.lang.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.native.*
import kotlinx.coroutines.*
import com.soywiz.korio.error.invalidOp as invalidOp1

class PromptConfigurator(
    private val _browser: Browser,
    private val emulator: Emulator,
    private val views: Views,
) {
    private val actions = linkedMapOf<String, suspend () -> Unit>()

    init {
        actions["help"] = { views.gameWindow.alert("Type any of the following actions: ${actions.keys}") }
        actions["dropbox"] = { dropbox() }
        actions["storage"] = { storage() }
        actions["memdump"] = { memdump() }
        actions["refresh"] = { KPspEmuNative.invalidateCache() }
    }

    suspend fun prompt() {
        try {
            //val action = browser.prompt("(F7) Action to perform ${actions.keys}", "")
            val action = views.gameWindow.prompt("(F7) Action to perform ${actions.keys}", "")
            println("action -> $action")
            if (action in actions) {
                actions[action]!!()
            } else {
                invalidOp1("Unknown action $action : Supported actions: ${actions.keys}")
            }
        } catch (e: CancellationException) {
            // Do nothing
        } catch (e: Throwable) {
            e.printStackTrace()
            views.gameWindow.alert("Error: ${e.message}\n\n$e")
        }
    }

    private suspend fun dropbox() {
        val bearerConfig = emulator.configManager.dropboxBearer
        val bearer = views.gameWindow.prompt("Set Dropbox Bearer token", bearerConfig.get())
        bearerConfig.set(bearer.trim())
    }

    private suspend fun storage() {
        val supportedStorages = listOf("local", "dropbox")
        val storageConfig = emulator.configManager.storage
        val storage = views.gameWindow.prompt("Select storage engine ($supportedStorages)", storageConfig.get())
        if (storage !in supportedStorages) invalidOp1("Invalid storage type '$storage' (must be )")
        storageConfig.set(storage)
    }

    private suspend fun memdump() {
        val outFile = applicationVfs["memdump.bin"]
        outFile.writeBytes(emulator.mem.readBytes(MemoryInfo.MAINMEM.start, MemoryInfo.MAINMEM.size))
        KpspemuMainScene.logger.warn { "Writted memory to $outFile" }
        views.gameWindow.alert("Writted memory to $outFile")
    }
}