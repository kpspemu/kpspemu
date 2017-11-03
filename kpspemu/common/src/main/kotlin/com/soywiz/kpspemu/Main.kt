package com.soywiz.kpspemu

import com.soywiz.klock.KLOCK_VERSION
import com.soywiz.korag.Korag
import com.soywiz.korau.Korau
import com.soywiz.korge.Korge
import com.soywiz.korge.bitmapfont.BitmapFont
import com.soywiz.korge.input.*
import com.soywiz.korge.render.RenderContext
import com.soywiz.korge.scene.Module
import com.soywiz.korge.scene.Scene
import com.soywiz.korge.scene.sleep
import com.soywiz.korge.service.Browser
import com.soywiz.korge.time.seconds
import com.soywiz.korge.tween.get
import com.soywiz.korge.tween.tween
import com.soywiz.korge.view.*
import com.soywiz.korim.color.RGBA
import com.soywiz.korim.font.BitmapFontGenerator
import com.soywiz.korio.JvmStatic
import com.soywiz.korio.Korio
import com.soywiz.korio.async.AsyncThread
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.inject.AsyncInjector
import com.soywiz.korio.lang.printStackTrace
import com.soywiz.korio.stream.openSync
import com.soywiz.korio.util.OS
import com.soywiz.korio.vfs.IsoVfs
import com.soywiz.korio.vfs.VfsFile
import com.soywiz.korio.vfs.applicationVfs
import com.soywiz.korma.Korma
import com.soywiz.korma.Matrix2d
import com.soywiz.korma.geom.Rectangle
import com.soywiz.korma.geom.SizeInt
import com.soywiz.korui.Korui
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

fun main(args: Array<String>) = Main.main(args)

object Main {
	@JvmStatic
	fun main(args: Array<String>) = Korge(KpspemuModule, injector = AsyncInjector()
		.mapPrototype(KpspemuMainScene::class) { KpspemuMainScene(get(Browser::class)) }
		.mapSingleton(Browser::class) { Browser(get(AsyncInjector::class)) }
	)
}

object KpspemuModule : Module() {
	//override val clearEachFrame: Boolean = false
	override val clearEachFrame: Boolean = true
	override val mainScene: KClass<out Scene> = KpspemuMainScene::class
	override val title: String = "kpspemu"
	override val size: SizeInt get() = SizeInt(480, 272)
}

class KpspemuMainScene(
	val browser: Browser
) : Scene(), WithEmulator {
	lateinit var exeFile: VfsFile
	lateinit override var emulator: Emulator
	val tex by lazy { views.texture(display.bmp) }
	val agRenderer by lazy { AGRenderer(this, tex) }
	val hudFont by lazy { BitmapFont(views.ag, "Lucida Console", 32, BitmapFontGenerator.LATIN_ALL, mipmaps = false) }
	var running = true
	var ended = false
	var paused = false

	suspend fun createEmulatorWithExe(exeFile: VfsFile) {
		running = true
		ended = false
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
	}

	lateinit var hud: Container

	suspend override fun sceneInit(sceneView: Container) {
		println("KPSPEMU: ${Kpspemu.VERSION}")
		println("KLOCK: $KLOCK_VERSION")
		println("KORMA: ${Korma.VERSION}")
		println("KORIO: ${Korio.VERSION}")
		println("KORAG: ${Korag.VERSION}")
		println("KORAU: ${Korau.VERSION}")
		println("KORUI: ${Korui.VERSION}")
		println("KORGE: ${Korge.VERSION}")

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
		//val exeFile = samplesFolder["cube.iso"]
		//val exeFile = samplesFolder["lights.pbp"]
		//val exeFile = samplesFolder["cwd.elf"]
		//val exeFile = samplesFolder["nehetutorial03.pbp"]
		//val exeFile = samplesFolder["polyphonic.elf"]
		//val exeFile = samplesFolder["text.elf"]
		//val exeFile = samplesFolder["cavestory.iso"]
		//val exeFile = samplesFolder["cavestory.zip"]
		//val exeFile = samplesFolder["TrigWars.iso"]
		val exeFile = samplesFolder["TrigWars.zip"]

		hud = views.container()
		hud += views.solidRect(96, 272, RGBA(0, 0, 0, 0xAA)).apply {
			enabled = false
			mouseEnabled = false
		}

		createEmulatorWithExe(exeFile)

		sceneView.addUpdatable {
			//controller.updateButton(PspCtrlButtons.cross, true) // auto press X
			if (!paused) {
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
						display.clear()
					}
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

		hud.alpha = 0.0

		fun getInfoText(): String = "kpspemu\n${Kpspemu.VERSION}\n\n${agRenderer.stats.toString()}"

		val infoText = views.text(getInfoText(), textSize = 10.0, font = hudFont).apply {
			x = 8.0
			y = 8.0
		}

		hud += infoText

		hud += views.simpleButton("Load...", font = hudFont).apply {
			x = 8.0
			y = 272.0 - 32.0 * 1
		}.onClick {
			createEmulatorWithExe(browser.openFile())
		}

		hud += views.simpleButton("Direct", font = hudFont).apply {
			x = 8.0
			y = 272.0 - 32.0 * 2
		}.onClick {
			agRenderer.directFastSharpRendering = !agRenderer.directFastSharpRendering
		}

		hud += views.simpleButton("Pause", font = hudFont).apply {
			x = 8.0
			y = 272.0 - 32.0 * 3
		}.onClick {
			paused = !paused
			it.view.setText(if (paused) "Resume" else "Pause")
		}

		val displayView = object : View(views) {
			override fun getLocalBoundsInternal(out: Rectangle): Unit = run { out.setTo(0, 0, 512, 272) }
			override fun render(ctx: RenderContext, m: Matrix2d) {
				agRenderer.render(ctx, m)
				infoText.text = getInfoText()
			}
		}

		sceneView += displayView
		sceneView += hud

		sceneView.onMove { hudOpen() }
		sceneView.onOut { hudClose() }
		sceneView.onClick { if (hud.alpha < 0.5) { hudOpen() } else { hudClose() } }
		sceneView.onKeyTyped { println(it.keyCode) }
		sceneView.onKeyDown { updateKey(it.keyCode, true) }
		sceneView.onKeyUp { updateKey(it.keyCode, false) }

	}

	val hudQueue = AsyncThread()

	suspend fun hudOpen() = hudQueue.cancelAndQueue {
		hud.tween(hud::alpha[1.0], hud::x[0.0], time = 0.2.seconds)
		sleep(2.seconds)
		hud.tween(hud::alpha[0.0], hud::x[-32.0], time = 0.2.seconds)
	}

	suspend fun hudClose() = hudQueue.cancelAndQueue {
		hud.tween(hud::alpha[0.0], hud::x[-32.0], time = 0.2.seconds)
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
						deviceManager.mount("ms0:/PSP/GAME/virtual", iso)
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

fun Views.simpleButton(text: String, width: Int = 80, height: Int = 24, font: BitmapFont = this.defaultFont): View {
	val button = container()
	val colorOver = RGBA(0xA0, 0xA0, 0xA0, 0xFF)
	val colorOut = RGBA(0x90, 0x90, 0x90, 0xFF)

	val bg = solidRect(width, height, colorOut)
	button += bg
	button += text(text, font = font).apply {
		x = 4.0
		y = 4.0
		enabled = false
	}
	button.onOut { bg.colorMul = colorOut }
	button.onOver { bg.colorMul = colorOver }
	return button
}