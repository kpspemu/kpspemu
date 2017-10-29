package com.soywiz.kpspemu

import com.soywiz.korge.Korge
import com.soywiz.korge.scene.Module
import com.soywiz.korge.scene.Scene
import com.soywiz.korge.view.Container
import com.soywiz.korge.view.image
import com.soywiz.korge.view.texture
import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.color.RGBA
import com.soywiz.korio.JvmStatic
import com.soywiz.korio.crypto.Base64
import com.soywiz.korio.inject.AsyncInjector
import com.soywiz.korio.stream.openSync
import com.soywiz.korio.stream.readAll
import com.soywiz.korio.stream.sliceWithSize
import com.soywiz.korma.geom.SizeInt
import com.soywiz.kpspemu.format.Elf
import com.soywiz.kpspemu.format.ElfPspModuleInfo
import com.soywiz.kpspemu.hle.modules.UtilsForUser
import com.soywiz.kpspemu.hle.modules.sceCtrl
import com.soywiz.kpspemu.hle.modules.sceDisplay
import com.soywiz.kpspemu.mem.Memory
import kotlin.reflect.KClass

object Main {
	@JvmStatic fun main(args: Array<String>) = Korge(KpspemuModule, injector = AsyncInjector()
		.mapPrototype(KpspemuMainScene::class) {  KpspemuMainScene() }
	)
}

object KpspemuModule : Module() {
	override val mainScene: KClass<out Scene> = KpspemuMainScene::class
	override val title: String = "kpspemu"
	override val size: SizeInt get() = SizeInt(480, 272)
}

val MinifireElf by lazy {
	Base64.decode(
		"f0VMRgEBAQAAAAAAAAAAAAIACAABAAAACACQCDwAAAA0AAAAATCiEDQAIAABACgA\n" +
			"AwACAAAAAAAAAAAAAQAAALAAAAAAAJAIAACQCNgBAADYAQAABwAAABAAAAABAAAA\n" +
			"AQAAAAIAAAAAAJAIsAAAANgBAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAMAAAAAAAAA\n" +
			"AAAAAIgCAAAXAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAEBVU5QAJAIEDwEAAQm\n" +
			"UAAFJhQABiQCAAc8AIAIPCFIAAAAAAAAAAAAAAAAAAAAAAAATBsIACEgQAAhKAAA\n" +
			"ITAAAMwbCAAhIAAATBwIACD2vSchIKADzC8IAKAIETywCBI8AEQQPCEgAADgAQUk\n" +
			"EAEGJIxOCAAhECACALYINAAAQKD//wgl/f8BBQEAQiQAtDQ2/wGTJiEgoAMMMAgA\n" +
			"AQCUJisYkwL7/2AUAACCov4BFSQhIKADDDAIAFkAAyQCAGAUGwBDAA0ABwASEAAA\n" +
			"EBgAAAEAYyRAGgMAIKBxACQQVQABAEIkIKCCAgAAlaJZAAgkALIpNgCySzb+ASol\n" +
			"AAIkkQECJZEhIIUAAgIlkSEghQABACWRISCFAIIgBAABAIBc//+EJAEAZKEBACkl\n" +
			"KxAqAfL/QBQBAGsl//8IJQL8ayXt/wAdAvwpJRAACDwmgAgCIRAAAiFAAAAhUEAC\n" +
			"ISAAAgACSyUAAEKRAACCrAAIgqwAEIKsAQBKJSsYSwH5/2AUBACEJAEACCVaAAIt\n" +
			"9P9AFAAQhCTQCaQnAQAFJAxUCADUCaiPAQAAVcw6CADMUQgAISAAAgACBSQDAAYk\n" +
			"AQAHJMxPCAAhQCACIYhAAiQAJAohkAABAC5yb2RhdGEuc2NlTW9kdWxlSW5mbwA="
	)
}

class KpspemuMainScene : Scene() {
	val elf = Elf.read(MinifireElf.openSync())
	val emu = Emulator(mem = Memory())
	val bmp = Bitmap32(512, 272)
	val temp = ByteArray(512 * 272 * 4)
	val tex by lazy { views.texture(bmp) }

	suspend override fun sceneInit(sceneView: Container) {
		emu.run {
			// Hardcoded as first example
			val ph = elf.programHeaders[0]
			val programBytes = elf.stream.sliceWithSize(ph.offset.toLong(), ph.fileSize.toLong()).readAll()
			mem.write(ph.virtualAddress, programBytes)
			val moduleInfo = ElfPspModuleInfo(elf.sectionHeadersByName[".rodata.sceModuleInfo"]!!.stream.clone())
			//interpreter.trace = true
			interpreter.trace = false

			UtilsForUser().registerPspModule(emu)
			sceCtrl().registerPspModule(emu)
			sceDisplay().registerPspModule(emu)

			cpu.GPR[29] = 0x0A000000 // stack
			cpu.setPC(0x08900008) // PC
		}

		sceneView.addUpdatable {
			emu.interpreter.steps(1000000)
			emu.mem.read(emu.display.address, temp, 0, temp.size)
			RGBA.decodeToBitmap32(bmp, temp)
			val bmpData = bmp.data
			for (n in 0 until bmpData.size) bmpData[n] = (bmpData[n] and 0x00FFFFFF) or 0xFF000000.toInt()
			//bmp.transformColor { (it and 0x00FFFFFF) or 0xFF000000.toInt() }
			//println(bmp.data.sum())
			tex.update(bmp)
			//sceneView.removeChildren()
			//sceneView += views.image(views.texture(bmp))
		}

		sceneView += views.image(tex)
	}
}