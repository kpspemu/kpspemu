package com.soywiz.kpspemu

import com.soywiz.klock.KLOCK_VERSION
import com.soywiz.klock.Klock
import com.soywiz.korag.Korag
import com.soywiz.korau.Korau
import com.soywiz.korge.Korge
import com.soywiz.korge.bitmapfont.BitmapFont
import com.soywiz.korge.input.*
import com.soywiz.korge.render.RenderContext
import com.soywiz.korge.scene.Module
import com.soywiz.korge.scene.Scene
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
import com.soywiz.korio.async.go
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.inject.AsyncInjector
import com.soywiz.korio.lang.printStackTrace
import com.soywiz.korio.stream.*
import com.soywiz.korio.util.OS
import com.soywiz.korio.util.umod
import com.soywiz.korio.vfs.*
import com.soywiz.korma.Korma
import com.soywiz.korma.Matrix2d
import com.soywiz.korma.geom.Rectangle
import com.soywiz.korma.geom.SizeInt
import com.soywiz.korui.Korui
import com.soywiz.kpspemu.ctrl.PspCtrlButtons
import com.soywiz.kpspemu.format.Pbp
import com.soywiz.kpspemu.format.elf.PspElf
import com.soywiz.kpspemu.format.elf.loadElfAndSetRegisters
import com.soywiz.kpspemu.format.openAsCso
import com.soywiz.kpspemu.ge.GeBatchData
import com.soywiz.kpspemu.ge.GpuRenderer
import com.soywiz.kpspemu.hle.registerNativeModules
import com.soywiz.kpspemu.mem.Memory
import com.soywiz.kpspemu.native.KPspEmuNative
import com.soywiz.kpspemu.util.asVfsFile
import com.soywiz.kpspemu.util.charset.ASCII
import com.soywiz.kpspemu.util.io.IsoVfs2
import com.soywiz.kpspemu.util.io.ZipVfs2
import kotlin.reflect.KClass

fun main(args: Array<String>) = Main.main(args)

object Main {
	@JvmStatic
	fun main(args: Array<String>) = Korge(KpspemuModule, injector = AsyncInjector()
		.mapPrototype(KpspemuMainScene::class) { KpspemuMainScene(get(Browser::class)) }
		.mapSingleton(Browser::class) { Browser(get(AsyncInjector::class)) }
		//, debug = true
	)
}

object KpspemuModule : Module() {
	//override val clearEachFrame: Boolean = false
	override val clearEachFrame: Boolean = true
	override val mainScene: KClass<out Scene> = KpspemuMainScene::class
	override val title: String = "kpspemu"
	override val size: SizeInt get() = SizeInt(480, 272)
	override val windowSize: SizeInt get() = SizeInt(480 * 2, 272 * 2)
}

class KpspemuMainScene(
	val browser: Browser
) : Scene(), WithEmulator {
	//lateinit var exeFile: VfsFile
	lateinit override var emulator: Emulator
	val tex by lazy { views.texture(display.bmp) }
	val agRenderer by lazy { AGRenderer(this, tex) }
	val hudFont by lazy { BitmapFont(views.ag, "Lucida Console", 32, BitmapFontGenerator.LATIN_ALL, mipmaps = false) }
	var running = true
	var ended = false
	var paused = false
	var forceSteps = 0

	suspend fun createEmulator(): Emulator {
		return Emulator(
			coroutineContext,
			mem = Memory(),
			gpuRenderer = object : GpuRenderer {
				override fun render(batches: List<GeBatchData>) {
					agRenderer.anyBatch = true
					agRenderer.batchesQueue += batches
				}
			}
		)
	}

	suspend fun createEmulatorWithExe(exeFile: VfsFile) {
		running = true
		ended = false
		emulator = createEmulator()
		agRenderer.anyBatch = false
		emulator.registerNativeModules()
		emulator.loadExecutableAndStart(exeFile)
	}

	lateinit var hud: Container

	suspend override fun sceneInit(sceneView: Container) {
		emulator = createEmulator()
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
		//val exeFile = samplesFolder["TrigWars.zip"]
		//val exeFile = samplesFolder["TrigWars.deflate.zip"]

		hud = views.container()
		//createEmulatorWithExe(exeFile)

		sceneView.addUpdatable {
			//controller.updateButton(PspCtrlButtons.cross, true) // auto press X
			if (!paused || forceSteps > 0) {
				if (forceSteps > 0) forceSteps--
				if (running && emulator.running) {
					val startTime = Klock.currentTimeMillis()
					try {
						emulator.frameStep()
					} catch (e: Throwable) {
						e.printStackTrace()
						running = false
					}
					val endTime = Klock.currentTimeMillis()
					agRenderer.stats.cpuTime = (endTime - startTime).toInt()
					agRenderer.updateStats()
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
				in 73..76 -> Unit // IJKL (analog)
				else -> println("UnhandledKey($pressed): $keyCode")
			}

			if (pressed) {
				when (keyCode) {
					121 -> { // F10
						go(coroutineContext) { toggleHud() }
					}
					122 -> { // F11
						// Dump threads
						for (thread in emulator.threadManager.threads) {
							println("Thread(${thread.name}) : ${thread.waitObject}")
						}
					}
				}
			}

			controller.updateAnalog(
				x = when { keys[74] -> -1f; keys[76] -> +1f; else -> 0f; },
				y = when { keys[73] -> -1f; keys[75] -> +1f; else -> 0f; }
			)
		}

		//hud.alpha = 0.0
		//hud.mouseEnabled = false

		fun getInfoText(): String {
			val out = arrayListOf<String>()
			out += "kpspemu"
			out += Kpspemu.VERSION
			out += ""
			out += "totalThreads=${emulator.threadManager.totalThreads}"
			out += "waitingThreads=${emulator.threadManager.waitingThreads}"
			out += "activeThreads=${emulator.threadManager.activeThreads}"
			out += ""
			out += agRenderer.stats.toString()
			out += ""
			out += "FPS: ${fpsCounter.getFpsInt()}"
			return out.joinToString("\n")
		}

		val infoText = views.text(getInfoText(), textSize = 8.0, font = hudFont).apply {
			x = 4.0
			y = 4.0
		}

		val loadButton = views.simpleButton("Load...", font = hudFont).apply {
			x = 8.0
			y = 272.0 - 32.0 * 1
		}.onClick {
			createEmulatorWithExe(browser.openFile())
		}

		val directButton = views.simpleButton("Auto", font = hudFont).apply {
			x = 8.0
			y = 272.0 - 32.0 * 2
		}.onClick {
			agRenderer.renderMode = when (agRenderer.renderMode) {
				AGRenderer.RenderMode.AUTO -> AGRenderer.RenderMode.NORMAL
				AGRenderer.RenderMode.NORMAL -> AGRenderer.RenderMode.AUTO
				else -> AGRenderer.RenderMode.AUTO
			//AGRenderer.RenderMode.AUTO -> AGRenderer.RenderMode.NORMAL
			//AGRenderer.RenderMode.NORMAL -> AGRenderer.RenderMode.DIRECT
			//AGRenderer.RenderMode.DIRECT -> AGRenderer.RenderMode.AUTO
			}

			it.view.setText(when (agRenderer.renderMode) {
				AGRenderer.RenderMode.AUTO -> "Auto"
				AGRenderer.RenderMode.NORMAL -> "Normal"
				AGRenderer.RenderMode.DIRECT -> "Direct"
			})
		}

		lateinit var pauseButton: View

		fun pause(set: Boolean) {
			paused = set
			pauseButton.setText(if (paused) "Resume" else "Pause")

		}

		pauseButton = views.simpleButton("Pause", font = hudFont).apply {
			x = 8.0
			y = 272.0 - 32.0 * 3
		}.onClick {
			pause(!paused)
		}!!

		val stepButton = views.simpleButton("Step", font = hudFont).apply {
			x = 8.0
			y = 272.0 - 32.0 * 4
		}.onClick {
			pause(true)
			forceSteps++
		}

		hud += views.solidRect(96, 272, RGBA(0, 0, 0, 0xCC)).apply { enabled = false; mouseEnabled = false }
		hud += infoText
		hud += loadButton
		hud += directButton
		hud += pauseButton
		hud += stepButton

		val displayView = object : View(views) {
			override fun getLocalBoundsInternal(out: Rectangle): Unit = run { out.setTo(0, 0, 512, 272) }
			override fun render(ctx: RenderContext, m: Matrix2d) {
				val startTime = Klock.currentTimeMillis()
				fpsCounter.tick(startTime.toDouble())
				agRenderer.render(views, ctx, m)
				val endTime = Klock.currentTimeMillis()
				agRenderer.stats.renderTime = (endTime - startTime).toInt()
				infoText.text = getInfoText()
			}
		}

		sceneView += displayView
		sceneView += hud

		//sceneView.onMove { hudOpen() }
		sceneView.onOut { hudClose() }
		sceneView.onClick { toggleHud() }
		//sceneView.onKeyTyped { println(it.keyCode) }
		sceneView.onKeyDown { updateKey(it.keyCode, true) }
		sceneView.onKeyUp { updateKey(it.keyCode, false) }

		if (OS.isBrowserJs) {
			val hash = KPspEmuNative.documentLocationHash.trim('#')
			val location = hash.trim('#').trim()
			println("Hash:$hash, Location:$location")
			if (location.isNotEmpty()) {
				hudCloseImmediate()
				createEmulatorWithExe(localCurrentDirVfs[location])
			}
		}
	}

	val fpsCounter = FpsCounter()
	val hudQueue = AsyncThread()

	suspend fun toggleHud() {
		if (hud.alpha < 0.5) {
			hudOpen()
		} else {
			hudClose()
		}
	}

	suspend fun hudOpen() = hudQueue.cancelAndQueue {
		hud.mouseEnabled = true
		hud.tween(hud::alpha[1.0], hud::x[0.0], time = 0.2.seconds)
		//sleep(2.seconds)
		//hud.tween(hud::alpha[0.0], hud::x[-32.0], time = 0.2.seconds)
	}

	suspend fun hudClose() = hudQueue.cancelAndQueue {
		hud.mouseEnabled = false
		hud.tween(hud::alpha[0.0], hud::x[-32.0], time = 0.2.seconds)
	}

	suspend fun hudCloseImmediate() = hudQueue.cancelAndQueue {
		hud.mouseEnabled = false
		hud.alpha = 0.0
		hud.x = -32.0
	}
}

class FpsCounter {
	val MAX_SAMPLES = 100
	var renderTimes = DoubleArray(MAX_SAMPLES)
	var renderTimeOffset = 0
	var renderCount = 0

	fun tick(time: Double) {
		renderTimes[renderTimeOffset++ % MAX_SAMPLES] = time
		if (renderCount < MAX_SAMPLES) renderCount++
	}

	fun getSample(offset: Int) = renderTimes[(renderTimeOffset - 1 - offset) umod MAX_SAMPLES]

	fun getFpsInt(): Int = getFps().toInt()

	fun getFps(): Double {
		if (renderCount == 0) return 0.0
		val elapsed = getSample(0) - getSample(renderCount - 1)
		return 1000.0 * (renderCount.toDouble() / elapsed)
	}
}

suspend fun Emulator.loadExecutableAndStart(file: VfsFile): PspElf {
	val PRELOAD_THRESHOLD = 3L * 1024 * 1024 // 3MB
	return loadExecutableAndStartInternal(when {
		file.size() < PRELOAD_THRESHOLD -> file.readAll().openAsync().asVfsFile(file.fullname)
		else -> file
	})
}

suspend fun Emulator.loadExecutableAndStartInternal(file: VfsFile): PspElf {
	return loadExecutableAndStartInternal(file, file.open().readBytesUpTo(0x10).openSync())
}

suspend fun Emulator.loadExecutableAndStartInternal(file: VfsFile, magic: SyncStream): PspElf {
	val magicId = magic.slice().readStringz(4, ASCII)
	val extension = when (magicId) {
		"\u007fELF" -> "elf"
		"\u0000PBP" -> "pbp"
		"CISO" -> "cso"
		else -> file.extensionLC
	}
	when (extension) {
		"elf", "prx", "bin" -> return loadElfAndSetRegisters(file.readAll().openSync())
		"pbp" -> return loadExecutableAndStartInternal(Pbp.load(file.open())[Pbp.PSP_DATA]!!.asVfsFile("executable.elf"))
		"cso", "ciso" -> return loadExecutableAndStartInternal(file.openAsCso().asVfsFile(file.pathInfo.pathWithExtension("cso.iso")))
		"iso", "zip" -> {
			val iso = when (extension) {
				"iso" -> IsoVfs2(file)
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
					return loadExecutableAndStartInternal(f)
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
	val txt = text(text, font = font).apply {
		this.x = 4.0
		this.y = 4.0
		this.autoSize = true
		//this.textBounds.setBounds(0, 0, width - 8, height - 8)
		//this.width = width - 8.0
		//this.height = height - 8.0
		//this.enabled = false
		//this.mouseEnabled = false
		//this.enabled = false
	}
	button += bg
	button += txt
	button.onOut { bg.colorMul = colorOut }
	button.onOver { bg.colorMul = colorOver }
	//txt.textBounds.setBounds(0, 0, 50, 50)
	return button
}