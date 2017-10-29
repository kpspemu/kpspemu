package com.soywiz.kpspemu

import com.soywiz.korge.Korge
import com.soywiz.korge.scene.Module
import com.soywiz.korge.scene.Scene
import com.soywiz.korge.view.Container
import com.soywiz.korge.view.image
import com.soywiz.korge.view.texture
import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korio.JvmStatic
import com.soywiz.korio.inject.AsyncInjector
import com.soywiz.korio.lang.printStackTrace
import com.soywiz.korio.stream.openSync
import com.soywiz.korio.util.OS
import com.soywiz.korio.vfs.applicationVfs
import com.soywiz.korma.geom.SizeInt
import com.soywiz.kpspemu.cpu.SP
import com.soywiz.kpspemu.format.elf.loadElfAndSetRegisters
import com.soywiz.kpspemu.hle.registerNativeModules
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

class KpspemuMainScene : Scene() {
	suspend override fun sceneInit(sceneView: Container) {
		val samplesFolder = when {
			OS.isJs -> applicationVfs
			//else -> ResourcesVfs
			else -> applicationVfs["samples"].jail()
		}

		//val elfBytes = samplesFolder["HelloWorldPSP.elf"].readAll()
		val elfBytes = samplesFolder["rtctest.elf"].readAll()

		val emu = Emulator(mem = Memory()).apply {
			registerNativeModules()
			loadElfAndSetRegisters(elfBytes.openSync())
		}
		val bmp = Bitmap32(512, 272)
		val tex by lazy { views.texture(bmp) }

		//emu.interpreter.trace = false
		emu.startThread.interpreter.trace = true

		var running = true

		sceneView.addUpdatable {
			emu.run {
				if (running) {
					try {
						frameStep()

						if (display.rawDisplay) {
							display.decodeToBitmap32(bmp)
							tex.update(bmp)
						}
					} catch (e: Throwable) {
						e.printStackTrace()
						running = false
					}
				}
			}

		}

		sceneView += views.image(tex)
	}
}