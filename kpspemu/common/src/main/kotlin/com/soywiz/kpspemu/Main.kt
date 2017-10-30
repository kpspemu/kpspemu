package com.soywiz.kpspemu

import com.soywiz.korge.Korge
import com.soywiz.korge.render.RenderContext
import com.soywiz.korge.scene.Module
import com.soywiz.korge.scene.Scene
import com.soywiz.korge.view.Container
import com.soywiz.korge.view.View
import com.soywiz.korge.view.image
import com.soywiz.korge.view.texture
import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.color.Colors
import com.soywiz.korio.JvmStatic
import com.soywiz.korio.inject.AsyncInjector
import com.soywiz.korio.lang.printStackTrace
import com.soywiz.korio.stream.openSync
import com.soywiz.korio.util.OS
import com.soywiz.korio.vfs.applicationVfs
import com.soywiz.korma.Matrix2d
import com.soywiz.korma.geom.SizeInt
import com.soywiz.kpspemu.format.elf.loadElfAndSetRegisters
import com.soywiz.kpspemu.ge.GeBatch
import com.soywiz.kpspemu.ge.GpuRenderer
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

class KpspemuMainScene : Scene(), WithEmulator {
	lateinit override var emulator: Emulator

	class KorgeRenderer(val scene: KpspemuMainScene) : View(scene.views), GpuRenderer, WithEmulator {
		override val emulator: Emulator get() = scene.emulator
		val batchesQueue = arrayListOf<List<GeBatch>>()

		override fun render(batches: List<GeBatch>) {
			batchesQueue += batches
			//println("KorgeRenderer.render: $batches")
			//display.rawDisplay = false
		}

		val tempBmp = Bitmap32(512, 272)

		override fun render(ctx: RenderContext, m: Matrix2d) {
			if (batchesQueue.isNotEmpty()) {
				try {
					val ag = ctx.ag
					ag.renderToBitmap(tempBmp) {
						ag.clear(Colors.RED) // @TODO: Remove this

						for (batches in batchesQueue) {
							for (batch in batches) {
								println(batch)
							}
						}
					}
					mem.write(display.address, tempBmp.data)
				} finally {
					batchesQueue.clear()
				}
			}
		}
	}

	suspend override fun sceneInit(sceneView: Container) {
		val samplesFolder = when {
			OS.isJs -> applicationVfs
		//else -> ResourcesVfs
			else -> applicationVfs["samples"].jail()
		}

		//val elfBytes = samplesFolder["minifire.elf"].readAll()
		//val elfBytes = samplesFolder["HelloWorldPSP.elf"].readAll()
		//val elfBytes = samplesFolder["rtctest.elf"].readAll()
		//val elfBytes = samplesFolder["compilerPerf.elf"].readAll()
		val elfBytes = samplesFolder["cube.elf"].readAll()

		val renderView = KorgeRenderer(this)

		emulator = Emulator(mem = Memory(), gpuRenderer = renderView).apply {
			registerNativeModules()
			loadElfAndSetRegisters(elfBytes.openSync())
			//threadManager.trace("_start")
			//threadManager.trace("user_main")
		}
		val tex by lazy { views.texture(display.bmp) }

		var running = true

		sceneView.addUpdatable {
				if (running) {
					try {
						emulator.frameStep()
					} catch (e: Throwable) {
						e.printStackTrace()
						running = false
					}

					if (display.rawDisplay) {
						display.decodeToBitmap32(display.bmp)
						tex.update(display.bmp)
					}
				}


		}

		sceneView += renderView
		sceneView += views.image(tex)
	}
}