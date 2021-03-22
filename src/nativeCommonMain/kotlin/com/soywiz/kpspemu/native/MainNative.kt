package com.soywiz.kpspemu.native

import com.soywiz.kpspemu.*

fun main(args: Array<String>) = Main.main(args)

/*
fun main(args: Array<String>) {
    Korio {
        val time = measureTime {
            var pspautotests: VfsFile = localCurrentDirVfs
            var rootTestResources: VfsFile = localCurrentDirVfs

            for (rootPath in listOf(
                ".",
                "..",
                "../..",
                "../../..",
                "../../../..",
                "../../../../..",
                "../../../../../..",
                "../../../../../../.."
            )) {
                //println("localCurrentDirVfs=$localCurrentDirVfs")
                //println("localCurrentDirVfs[rootPath]=${localCurrentDirVfs[rootPath]}")
                val root = localCurrentDirVfs[rootPath].jail()
                pspautotests = root["pspautotests"]
                rootTestResources = root["kpspemu/common/testresources"]
                if (pspautotests.exists()) {
                    break
                }
            }
            //val elf = localCurrentDirVfs["game.prx"].readAsSyncStream()
            val elf = pspautotests["cpu/cpu_alu/cpu_alu.prx"].readAsSyncStream()

            // Run test in release mode for benchmarking
            val emulator = Emulator(kotlin.coroutines.experimental.coroutineContext)
            emulator.interpreted = true
            emulator.display.exposeDisplay = false
            emulator.registerNativeModules()
            //val info = emulator.loadElfAndSetRegisters(elf, "ms0:/PSP/GAME/EBOOT.PBP")
            emulator.fileManager.currentDirectory = "ms0:/PSP/GAME/virtual"
            emulator.fileManager.executableFile = "ms0:/PSP/GAME/virtual/EBOOT.PBP"
            emulator.deviceManager.mount(
                emulator.fileManager.currentDirectory,
                MemoryVfsMix("EBOOT.PBP" to elf.clone().toAsync())
            )
            val info = emulator.loadElfAndSetRegisters(elf, listOf("ms0:/PSP/GAME/virtual/EBOOT.PBP"))

            var generatedError: Throwable? = null

            try {
                //println("[1]")
                while (emulator.running) {
                    //println("[2] : ${emulator.running}")
                    emulator.threadManager.step() // UPDATE THIS
                    getCoroutineContext().eventLoop.step(10)
                    //println("[3]")
                }
            } catch (e: Throwable) {
                Console.error("Partial output generated:")
                Console.error("'" + emulator.output.toString() + "'")
                //throw e
                e.printStackTrace()
                generatedError = e
            }

            println(emulator.output.toString())
            //assertEquals(expected.normalize(), processor(emulator.output.toString().normalize()))
            //generatedError?.let { throw it }
        }
        println("TIME: ${time.time}")
    }
}
*/
