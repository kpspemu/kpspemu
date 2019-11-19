package com.soywiz.kpspemu

import com.soywiz.klock.*
import com.soywiz.korge.util.*
import com.soywiz.korio.async.*
import com.soywiz.korio.file.*
import com.soywiz.korio.file.std.*

open class BaseTest {
    fun pspSuspendTest(callback: suspend Resources.() -> Unit) {
        suspendTest {
            withTimeout(20.seconds) {
                callback(Resources.apply { initOnce() })
            }
        }
    }

    @NativeThreadLocal
    object Resources {
        private var initialized = false
        lateinit var pspautotests: VfsFile
        lateinit var rootTestResources: VfsFile

        suspend fun initOnce() {
            if (!initialized) {
                initialized = true
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
                        return
                    }
                }
                error("Can't find root folder with 'pspautotests' and 'testresources'")
            }
        }
    }
}