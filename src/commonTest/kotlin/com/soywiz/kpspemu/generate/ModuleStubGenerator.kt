package com.soywiz.kpspemu.generate

import com.soywiz.korio.async.*
import com.soywiz.korio.file.std.*
import com.soywiz.korio.lang.*
import com.soywiz.korio.serialization.xml.*
import com.soywiz.korio.util.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.hle.psplibdoc.*
import com.soywiz.krypto.encoding.*
import kotlin.test.*

class ModuleStubGenerator : BaseTest() {
    @Ignore
    @Test
    fun generateModules() = suspendTest {
        val root = localCurrentDirVfs["../../psplibdoc"]
        val doc = LibDoc.parse(root["psplibdoc_660.xml"].readXml())
        //doc.dump()

        val registerNativeModules = Indenter.gen {
            line("fun ModuleManager.registerNativeModules() {")
            for (library in doc.allLibraries) {
                line("	register(${library.name}(emulator))")
            }
            line(")}")
        }

        println(registerNativeModules)
        //if (true) return@syncTest

        for (prx in doc.prxs) {
            for (library in prx.libraries) {
                val libraryFile = Indenter.gen {
                    line("package com.soywiz.kpspemu.hle.modules")
                    line("")
                    line("import com.soywiz.kpspemu.Emulator")
                    line("import com.soywiz.kpspemu.cpu.CpuState")
                    line("import com.soywiz.kpspemu.hle.SceModule")
                    line("")
                    line("@Suppress(\"UNUSED_PARAMETER\")")
                    line("class ${library.name}(emulator: Emulator) : SceModule(emulator, ${library.name.quote()}, ${library.flags.hex}, ${prx.fileName.quote()}, ${prx.name.quote()}) {")
                    for (function in library.functions) {
                        line("	fun ${function.name}(cpu: CpuState): Unit = UNIMPLEMENTED(${function.nid.hex})")
                    }
                    line("")
                    line("	override fun registerModule() {")
                    for (function in library.functions) {
                        line("		registerFunctionRaw(${function.name.quoted}, ${function.nid.hex}, since = 150) { ${function.name}(it) }")
                    }
                    line("	}")
                    line("}")
                }
                root["reference/${library.name}.kt"].writeString(libraryFile.toString())
            }
        }
    }
}