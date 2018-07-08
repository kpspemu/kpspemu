package com.soywiz.kpspemu

import com.soywiz.korio.async.*
import com.soywiz.korio.file.*
import com.soywiz.korio.file.std.*
import com.soywiz.std.*

open class BaseTest {
    fun pspSuspendTest(callback: suspend Resources.() -> Unit) {
        suspendTest { callback(Resources.apply { initOnce() }) }
    }

    @ThreadLocal
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