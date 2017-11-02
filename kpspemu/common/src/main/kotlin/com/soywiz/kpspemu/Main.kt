package com.soywiz.kpspemu

import com.soywiz.korge.Korge
import com.soywiz.korge.bitmapfont.BitmapFont
import com.soywiz.korge.input.onKeyDown
import com.soywiz.korge.input.onKeyTyped
import com.soywiz.korge.input.onKeyUp
import com.soywiz.korge.render.RenderContext
import com.soywiz.korge.scene.Module
import com.soywiz.korge.scene.Scene
import com.soywiz.korge.view.Container
import com.soywiz.korge.view.View
import com.soywiz.korge.view.text
import com.soywiz.korge.view.texture
import com.soywiz.korim.font.BitmapFontGenerator
import com.soywiz.korio.JvmStatic
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.inject.AsyncInjector
import com.soywiz.korio.lang.printStackTrace
import com.soywiz.korio.stream.openSync
import com.soywiz.korio.util.OS
import com.soywiz.korio.vfs.IsoVfs
import com.soywiz.korio.vfs.VfsFile
import com.soywiz.korio.vfs.applicationVfs
import com.soywiz.korma.Matrix2d
import com.soywiz.korma.geom.SizeInt
import com.soywiz.kpspemu.ctrl.PspCtrlButtons
import com.soywiz.kpspemu.format.Pbp
import com.soywiz.kpspemu.format.elf.PspElf
import com.soywiz.kpspemu.format.elf.loadElfAndSetRegisters
import com.soywiz.kpspemu.ge.GeBatch
import com.soywiz.kpspemu.ge.GpuRenderer
import com.soywiz.kpspemu.hle.registerNativeModules
import com.soywiz.kpspemu.mem.Memory
import com.soywiz.kpspemu.util.asVfsFile
import com.soywiz.kpspemu.util.io.ZipVfs2
import kotlin.reflect.KClass

const val DIRECT_FAST_SHARP_RENDERING = false
//const val DIRECT_FAST_SHARP_RENDERING = true

fun main(args: Array<String>) = Main.main(args)

object Main {
	@JvmStatic
	fun main(args: Array<String>) = Korge(KpspemuModule, injector = AsyncInjector()
		.mapPrototype(KpspemuMainScene::class) { KpspemuMainScene() }
	)
}

object KpspemuModule : Module() {
	//override val clearEachFrame: Boolean = false
	override val clearEachFrame: Boolean = true
	override val mainScene: KClass<out Scene> = KpspemuMainScene::class
	override val title: String = "kpspemu"
	override val size: SizeInt get() = SizeInt(480, 272)
}

class KpspemuMainScene : Scene(), WithEmulator {
	lateinit override var emulator: Emulator
	val tex by lazy { views.texture(display.bmp) }
	val agRenderer by lazy { AGRenderer(this, tex) }
	val hudFont by lazy { BitmapFont(views.ag, "Lucida Console", 32, BitmapFontGenerator.LATIN_ALL, mipmaps = false) }

	suspend override fun sceneInit(sceneView: Container) {
		val samplesFolder = when {
			OS.isJs -> applicationVfs
		//else -> ResourcesVfs
			else -> applicationVfs["samples"].jail()
		}

		//val exeFile = samplesFolder["minifire.elf"]
		//val exeFile = samplesFolder["HelloWorldPSP.elf"]
		//val exeFile = samplesFolder["rtctest.elf"]
		//val exeFile = samplesFolder["compilerPerf.elf"]
		//val exeFile = samplesFolder["cube.elf"]
		//val exeFile = samplesFolder["ortho.elf"]
		//val exeFile = samplesFolder["mytest.elf"]
		//val exeFile = samplesFolder["counter.elf"]
		//val exeFile = samplesFolder["controller.elf"]
		//val exeFile = samplesFolder["fputest.elf"]
		//val exeFile = samplesFolder["lines.elf"]
		//val exeFile = samplesFolder["lines.pbp"]
		//val exeFile = samplesFolder["polyphonic.elf"]
		val exeFile = samplesFolder["cube.iso"]
		//val exeFile = samplesFolder["lights.pbp"]
		//val exeFile = samplesFolder["cwd.elf"]
		//val exeFile = samplesFolder["nehetutorial03.pbp"]
		//val exeFile = samplesFolder["polyphonic.elf"]
		//val exeFile = samplesFolder["text.elf"]
		//val exeFile = samplesFolder["cavestory.iso"]
		//val exeFile = samplesFolder["cavestory.zip"]
		//val exeFile = samplesFolder["TrigWars.iso"]
		//val exeFile = samplesFolder["TrigWars.zip"]

		emulator = Emulator(
			coroutineContext,
			mem = Memory(),
			gpuRenderer = object : GpuRenderer {
				override fun render(batches: List<GeBatch>) {
					agRenderer.batchesQueue += batches
				}
			}
		).apply {
			registerNativeModules()
			loadExecutableAndStart(exeFile)
			//threadManager.trace("_start")
			//threadManager.trace("user_main")
		}

		var running = true
		var ended = false
		val hud = views.container()

		sceneView.addUpdatable {
			//controller.updateButton(PspCtrlButtons.cross, true) // auto press X

			if (running && emulator.running) {
				try {
					emulator.frameStep()
				} catch (e: Throwable) {
					e.printStackTrace()
					running = false
				}
			} else {
				if (!ended) {
					ended = true
					println("COMPLETED")
				}
			}
		}

		val keys = BooleanArray(256)

		fun updateKey(keyCode: Int, pressed: Boolean) {
			//println("updateKey: $keyCode, $pressed")
			keys[keyCode and 0xFF] = pressed
			when (keyCode) {
				10 -> controller.updateButton(PspCtrlButtons.start, pressed) // return
				32 -> controller.updateButton(PspCtrlButtons.select, pressed) // space
				87 -> controller.updateButton(PspCtrlButtons.triangle, pressed) // W
				65 -> controller.updateButton(PspCtrlButtons.square, pressed) // A
				83 -> controller.updateButton(PspCtrlButtons.cross, pressed) // S
				68 -> controller.updateButton(PspCtrlButtons.circle, pressed) // D
				81 -> controller.updateButton(PspCtrlButtons.leftTrigger, pressed) // Q
				69 -> controller.updateButton(PspCtrlButtons.rightTrigger, pressed) // E
				37 -> controller.updateButton(PspCtrlButtons.left, pressed) // LEFT
				38 -> controller.updateButton(PspCtrlButtons.up, pressed) // UP
				39 -> controller.updateButton(PspCtrlButtons.right, pressed) // RIGHT
				40 -> controller.updateButton(PspCtrlButtons.down, pressed) // DOWN
				113 -> { // F2
					if (pressed) hud.visible = !hud.visible
				}
				in 73..76 -> Unit // IJKL (analog)
				else -> println("UnhandledKey($pressed): $keyCode")
			}

			controller.updateAnalog(
				x = when { keys[74] -> -1f; keys[76] -> +1f; else -> 0f; },
				y = when { keys[73] -> +1f; keys[75] -> -1f; else -> 0f; }
			)
		}

		sceneView.onKeyTyped { println(it.keyCode) }
		sceneView.onKeyDown { updateKey(it.keyCode, true) }
		sceneView.onKeyUp { updateKey(it.keyCode, false) }

		val statsText = views.text(agRenderer.stats.toString(), textSize = 10.0, font = hudFont).apply {
			x = 8.0
			y = 8.0
		}

		hud += statsText

		sceneView += object : View(views) {
			override fun render(ctx: RenderContext, m: Matrix2d) {
				agRenderer.render(ctx, m)
				statsText.text = agRenderer.stats.toString()
			}
		}
		sceneView += hud
	}
}

suspend fun Emulator.loadExecutableAndStart(file: VfsFile): PspElf {
	when (file.extensionLC) {
		"elf", "prx", "bin" -> return loadElfAndSetRegisters(file.readAll().openSync())
		"pbp" -> return loadExecutableAndStart(Pbp.load(file.open())[Pbp.PSP_DATA]!!.asVfsFile("executable.elf"))
		"iso", "zip" -> {
			val iso = when (file.extensionLC) {
				"iso" -> IsoVfs(file)
				"zip" -> ZipVfs2(file.open(), file)
				else -> invalidOp("UNEXPECTED")
			}
			val paramSfo = iso["PSP_GAME/PARAM.SFO"]

			val files = listOf(
				iso["PSP_GAME/SYSDIR/BOOT.BIN"],
				iso["EBOOT.ELF"],
				iso["EBOOT.PBP"]
			)

			for (f in files) {
				if (f.exists()) {
					if (f.parent.path.isEmpty()) {
						fileManager.currentDirectory = "umd0:/"
						deviceManager.mount(fileManager.currentDirectory, iso)
						deviceManager.mount("game0:/", iso)
						deviceManager.mount("umd0:/", iso)
					}
					return loadExecutableAndStart(f)
				}
			}
			invalidOp("Can't find any possible executalbe in ISO ($files)")
		}
		else -> {
			invalidOp("Don't know how to load executable file $file")
		}
	}
}
