package com.soywiz.kpspemu.hle.manager

import com.soywiz.klock.*
import com.soywiz.korio.async.*
import com.soywiz.korio.file.*
import com.soywiz.korio.file.std.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.util.*
import com.soywiz.kpspemu.util.dropbox.*
import com.soywiz.kpspemu.util.io.*

class DeviceManager(val emulator: Emulator) {
    lateinit var ms: VfsFile
    val flash = MemoryVfsMix(
    )
    val dummy = MemoryVfs()

    val root = MountableVfsSyncNew {
    }

    val mountable = root.vfs as MountableSync

    suspend fun init() {
        reset()
    }

    suspend fun reset() {
        mountable.unmountAll()
    }

    //val devicesToVfs = LinkedHashMap<String, VfsFile>()

    fun mount(name: String, vfs: VfsFile) {
        mountable.unmount(name)
        mountable.mount(name, vfs)
        //devicesToVfs[name] = vfs
    }

    private var lastStorageType = ""
    private suspend fun updatedStorage() {
        val storage = emulator.configManager.storage.get()
        println("updatedStorage: $storage")
        if (storage == lastStorageType) {
            println("Already using that storage!")
            return
        }
        lastStorageType = storage

        val base: VfsFile = when (storage) {
            "local" -> ApplicationDataVfs["ms0"]
            "dropbox" -> DropboxVfs(Dropbox(emulator.configManager.dropboxBearer.get())).root
            else -> ApplicationDataVfs["ms0"]
        }
        ms = base.jail()
        mount("fatms0:", ms)
        mount("ms0:", ms)
        mount("mscmhc0:", ms)
        mount("host0:", dummy)
        mount("flash0:", flash)
        mount("emulator:", dummy)
        mount("kemulator:", dummy)
        mount("disc0:", dummy)
        mount("umd0:", dummy)

        println("Making directories...")
        launchImmediately(emulator.coroutineContext) {
            emulator.coroutineContext.delay(10.milliseconds)
            base.apply { mkdirsSafe() }
            ms["PSP"].mkdirsSafe()
            ms["PSP/GAME"].mkdirsSafe()
            ms["PSP/SAVES"].mkdirsSafe()
            println("Done")
        }
        println("Continuing...")
    }

    fun setStorage(storage: String) {
        println("Using storage: $storage")
        launchImmediately(emulator.coroutineContext) {
            updatedStorage()
        }
    }
}