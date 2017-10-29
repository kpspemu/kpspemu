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
import com.soywiz.kpspemu.cpu.SP
import com.soywiz.kpspemu.format.elf.Elf
import com.soywiz.kpspemu.format.elf.ElfPspModuleInfo
import com.soywiz.kpspemu.format.elf.loadElf
import com.soywiz.kpspemu.format.elf.loadElfAndSetRegisters
import com.soywiz.kpspemu.hle.modules.UtilsForUser
import com.soywiz.kpspemu.hle.modules.registerNativeModules
import com.soywiz.kpspemu.hle.modules.sceCtrl
import com.soywiz.kpspemu.hle.modules.sceDisplay
import com.soywiz.kpspemu.mem.Memory
import kotlin.reflect.KClass

fun main(args: Array<String>) = Main.main(args)

object Main {
	@JvmStatic
	fun main(args: Array<String>) = Korge(KpspemuModule, injector = AsyncInjector()
		.mapPrototype(KpspemuMainScene::class) { KpspemuMainScene() }
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
	val emu = Emulator(mem = Memory()).apply {
		registerNativeModules()
		loadElfAndSetRegisters(MinifireElf.openSync())
		cpu.SP = 0x0A000000 // @TODO: hardcoded stack
	}
	val bmp = Bitmap32(512, 272)
	val temp = ByteArray(512 * 272 * 4)
	val tex by lazy { views.texture(bmp) }

	suspend override fun sceneInit(sceneView: Container) {
		emu.interpreter.trace = false

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