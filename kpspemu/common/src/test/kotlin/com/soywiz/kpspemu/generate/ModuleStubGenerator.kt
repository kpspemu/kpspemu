package com.soywiz.kpspemu.generate

import com.soywiz.korio.async.syncTest
import com.soywiz.korio.serialization.xml.readXml
import com.soywiz.korio.util.Indenter
import com.soywiz.korio.util.quote
import com.soywiz.korio.vfs.localCurrentDirVfs
import com.soywiz.kpspemu.hle.psplibdoc.LibDoc
import com.soywiz.kpspemu.util.hex
import com.soywiz.kpspemu.util.quoted
import org.junit.Test
import kotlin.test.Ignore

class ModuleStubGenerator {
	@Ignore
	@Test
	fun generateModules() = syncTest {
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