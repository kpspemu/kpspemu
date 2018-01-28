package com.soywiz.kpspemu

import com.soywiz.dynarek.Dynarek
import com.soywiz.kds.umod
import com.soywiz.klock.Klock
import com.soywiz.klogger.Klogger
import com.soywiz.klogger.Logger
import com.soywiz.kmem.Kmem
import com.soywiz.korag.Korag
import com.soywiz.korau.Korau
import com.soywiz.korge.Korge
import com.soywiz.korge.bitmapfont.BitmapFont
import com.soywiz.korge.input.onClick
import com.soywiz.korge.input.onKeyDown
import com.soywiz.korge.input.onKeyUp
import com.soywiz.korge.render.RenderContext
import com.soywiz.korge.render.Texture
import com.soywiz.korge.scene.*
import com.soywiz.korge.service.Browser
import com.soywiz.korge.time.milliseconds
import com.soywiz.korge.time.seconds
import com.soywiz.korge.tween.get
import com.soywiz.korge.tween.tween
import com.soywiz.korge.view.*
import com.soywiz.korim.Korim
import com.soywiz.korim.bitmap.Bitmap
import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.color.RGBA
import com.soywiz.korim.font.BitmapFontGenerator
import com.soywiz.korim.format.PNG
import com.soywiz.korim.vector.Context2d
import com.soywiz.korinject.AsyncInjector
import com.soywiz.korinject.Korinject
import com.soywiz.korio.JvmStatic
import com.soywiz.korio.Korio
import com.soywiz.korio.async.AsyncThread
import com.soywiz.korio.async.Promise
import com.soywiz.korio.async.go
import com.soywiz.korio.coroutine.getCoroutineContext
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.lang.printStackTrace
import com.soywiz.korio.stream.AsyncStream
import com.soywiz.korio.stream.openAsync
import com.soywiz.korio.stream.openSync
import com.soywiz.korio.stream.readAll
import com.soywiz.korio.util.OS
import com.soywiz.korio.util.hex
import com.soywiz.korio.util.hexString
import com.soywiz.korio.vfs.*
import com.soywiz.korma.Korma
import com.soywiz.korma.Matrix2d
import com.soywiz.korma.geom.Rectangle
import com.soywiz.korma.geom.SizeInt
import com.soywiz.korui.Korui
import com.soywiz.kpspemu.ctrl.PspCtrlButtons
import com.soywiz.kpspemu.format.*
import com.soywiz.kpspemu.format.elf.CryptedElf
import com.soywiz.kpspemu.format.elf.PspElf
import com.soywiz.kpspemu.format.elf.loadElfAndSetRegisters
import com.soywiz.kpspemu.ge.GeBatchData
import com.soywiz.kpspemu.ge.GpuRenderer
import com.soywiz.kpspemu.hle.registerNativeModules
import com.soywiz.kpspemu.native.KPspEmuNative
import com.soywiz.kpspemu.ui.PromptConfigurator
import com.soywiz.kpspemu.ui.simpleButton
import com.soywiz.kpspemu.util.PspEmuKeys
import com.soywiz.kpspemu.util.io.openAsIso2
import com.soywiz.kpspemu.util.io.openAsZip2
import com.soywiz.kpspemu.util.mkdirsSafe
import kotlin.math.roundToInt
import kotlin.reflect.KClass

fun main(args: Array<String>) = Main.main(args)

object Main {
    @JvmStatic
    fun main(args: Array<String>) {
        Korge(KpspemuModule, injector = AsyncInjector()
            .mapSingleton(Emulator::class) { Emulator(getCoroutineContext()) }
            .mapSingleton(PromptConfigurator::class) {
                PromptConfigurator(
                    get(Browser::class),
                    get(Emulator::class)
                )
            }
            .mapPrototype(KpspemuMainScene::class) {
                KpspemuMainScene(
                    get(Browser::class),
                    get(Emulator::class),
                    get(PromptConfigurator::class)
                )
            }
            .mapPrototype(DebugScene::class) { DebugScene(get(Browser::class), get(Emulator::class)) }
            .mapSingleton(Browser::class) { Browser(get(AsyncInjector::class)) }

            // @TODO: Kotlin.JS unresolved bug!
            //.mapSingleton { Emulator(getCoroutineContext()) }
            //.mapSingleton { PromptConfigurator(get(), get()) }
            //.mapPrototype { KpspemuMainScene(get(), get(), get()) }
            //.mapPrototype { DebugScene(get(), get()) }
            //.mapSingleton { Browser(get()) }
        )
    }
}

object KpspemuModule : Module() {
    //override val clearEachFrame: Boolean = false

    override val clearEachFrame: Boolean = true
    override val mainScene: KClass<out Scene> = KpspemuMainScene::class
    override val title: String = "kpspemu - ${Kpspemu.VERSION}"
    override val iconImage: Context2d.SizedDrawable? = com.soywiz.korim.vector.format.SVG(KpspemuAssets.LOGO)
    override val size: SizeInt get() = SizeInt(480, 272)
    override val windowSize: SizeInt get() = SizeInt(480 * 2, 272 * 2)
}

abstract class SceneWithProcess() : Scene() {
    private val processes = arrayListOf<Promise<Unit>>()

    fun registerSceneProcess(callback: suspend () -> Unit) {
        processes += go { callback() }
    }

    override suspend fun sceneDestroy() {
        super.sceneDestroy()
        for (process in processes) process.cancel()
    }
}

class KpspemuMainScene(
    val browser: Browser,
    override val emulator: Emulator,
    val promptConfigurator: PromptConfigurator
) : SceneWithProcess(), WithEmulator {
    companion object {
        val logger = Logger("KpspemuMainScene")
        val MS_0 = 0.milliseconds
        val MS_1 = 1.milliseconds
        val MS_2 = 2.milliseconds
        val MS_10 = 10.milliseconds
        val MS_15 = 15.milliseconds
    }

    //lateinit var exeFile: VfsFile
    val tex by lazy { views.texture(display.bmp) }
    val agRenderer by lazy { AGRenderer(this, tex) }
    val hudFont by lazy { BitmapFont(views.ag, "Lucida Console", 32, BitmapFontGenerator.LATIN_ALL, mipmaps = false) }
    var running = true
    var ended = false
    var paused = false
    var forceSteps = 0

    suspend fun resetEmulator() {
        running = true
        ended = false
        emulator.reset()
        emulator.gpuRenderer = object : GpuRenderer() {
            override val queuedJobs get() = agRenderer.taskCountFlag

            override fun render(batches: List<GeBatchData>) {
                agRenderer.addBatches(batches)
            }

            override fun reset() {
                super.reset()
                agRenderer.reset()
            }

            private val identity = Matrix2d()
            private val renderContext = RenderContext(views.ag)

            override fun tryExecuteNow() {
                // @TODO: This produces an error! We should update KorAG to support this!
                //println("tryExecuteNow.Thread: ${KorioNative.currentThreadId}")
                views.ag.offscreenRendering {
                    agRenderer.render(views, renderContext, identity)
                }
            }
        }
        agRenderer.anyBatch = false
        agRenderer.reset()
        setIcon0Bitmap(Bitmap32(1, 1))
    }

    var icon0Texture: Texture? = null
    val icon0Image: Image by lazy {
        views.image(views.transparentTexture).apply {
            alpha = 0.5
        }
    }
    val titleText by lazy {
        views.text("").apply {
            alpha = 0.5
        }
    }

    fun setIcon0Bitmap(bmp: Bitmap) {
        icon0Image.tex = views.texture(bmp)
        icon0Texture?.close()
        icon0Texture = icon0Image.tex
    }

    suspend fun createEmulatorWithExe(exeFile: VfsFile) {
        setIcon0Bitmap(Bitmap32(144, 80))
        titleText.text = ""

        resetEmulator()
        emulator.registerNativeModules()
        emulator.loadExecutableAndStart(exeFile, object : LoadProcess() {
            suspend override fun readIcon0(icon0: ByteArray) {
                //icon0.writeToFile("c:/temp/icon0.png")
                try {
                    setIcon0Bitmap(PNG.read(icon0, "file.png").toBMP32())
                } catch (e: Throwable) {
                    println("ERROR at loadExecutableAndStart")
                    e.printStackTrace()
                }
            }

            suspend override fun readParamSfo(psf: Psf) {
                logger.warn { "PARAM.SFO:" }
                for ((key, value) in psf.entriesByName) {
                    logger.warn { "PSF: $key = $value" }
                }
                titleText.text = psf.entriesByName["TITLE"]?.toString() ?: ""
            }
        })
    }

    lateinit var hud: Container

    suspend fun cpuProcess() {
        while (true) {
            if (!paused || forceSteps > 0) {
                if (forceSteps > 0) forceSteps--
                if (running && emulator.running) {
                    val startTime = Klock.currentTimeMillis()
                    try {
                        emulator.threadManager.step()
                    } catch (e: Throwable) {
                        println("ERROR at cpuProcess")
                        e.printStackTrace()
                        running = false
                    }
                    val endTime = Klock.currentTimeMillis()
                    agRenderer.stats.cpuTime = (endTime - startTime).toInt()
                } else {
                    if (!ended) {
                        ended = true
                        println("COMPLETED")
                        display.crash()
                    }
                }
                emulator.threadManager.waitThreadChange()
            } else {
                sleep(MS_10)
            }
        }
    }

    suspend fun displayProcess() {
        while (true) {
            controller.startFrame(timeManager.getTimeInMicrosecondsInt())

            emulator.interruptManager.dispatchVsync()
            //sceneView.waitFrame()
            sleep(MS_15)

            /*
            //sleep(MS_15)
            sleep(MS_2)
            emulator.interruptManager.dispatchVsync()
            emulator.display.startVsync()
            controller.endFrame()
            sleep(MS_1)
            emulator.display.endVsync()
            */
            controller.endFrame()
        }
    }

    suspend override fun sceneInit(sceneView: Container) {
        registerSceneProcess { cpuProcess() }
        registerSceneProcess { displayProcess() }
        //val func = function(DClass(CpuState::class), DVOID) {
        //	SET(p0[CpuState::_PC], 7.lit)
        //}.generateDynarek()
        //
        //val cpuState = CpuState(GlobalCpuState(), Memory())
        //val result = func(cpuState)
        //println("PC: ${cpuState.PC.hex}")

        resetEmulator()
        println("KPSPEMU: ${Kpspemu.VERSION}")
        println("DYNAREK: ${Dynarek.VERSION}")
        println("KORINJECT: ${Korinject.VERSION}")
        println("KLOGGER: ${Klogger.VERSION}")
        println("KMEM: ${Kmem.VERSION}")
        println("KLOCK: ${Klock.VERSION}")
        println("KORMA: ${Korma.VERSION}")
        println("KORIO: ${Korio.VERSION}")
        println("KORIM: ${Korim.VERSION}")
        println("KORAG: ${Korag.VERSION}")
        println("KORAU: ${Korau.VERSION}")
        println("KORUI: ${Korui.VERSION}")
        println("KORGE: ${Korge.VERSION}")

        hud = views.container()
        //createEmulatorWithExe(exeFile)

        val keys = BooleanArray(256)

        fun updateKey(keyCode: Int, pressed: Boolean) {
            //println("updateKey: $keyCode, $pressed")
            keys[keyCode and 0xFF] = pressed
            when (keyCode) {
                PspEmuKeys.RETURN -> controller.updateButton(PspCtrlButtons.start, pressed)
                PspEmuKeys.SPACE -> controller.updateButton(PspCtrlButtons.select, pressed)
                PspEmuKeys.W -> controller.updateButton(PspCtrlButtons.triangle, pressed)
                PspEmuKeys.A -> controller.updateButton(PspCtrlButtons.square, pressed)
                PspEmuKeys.S -> controller.updateButton(PspCtrlButtons.cross, pressed)
                PspEmuKeys.D -> controller.updateButton(PspCtrlButtons.circle, pressed)
                PspEmuKeys.Q -> controller.updateButton(PspCtrlButtons.leftTrigger, pressed)
                PspEmuKeys.E -> controller.updateButton(PspCtrlButtons.rightTrigger, pressed)
                PspEmuKeys.LEFT -> controller.updateButton(PspCtrlButtons.left, pressed)
                PspEmuKeys.UP -> controller.updateButton(PspCtrlButtons.up, pressed)
                PspEmuKeys.RIGHT -> controller.updateButton(PspCtrlButtons.right, pressed)
                PspEmuKeys.DOWN -> controller.updateButton(PspCtrlButtons.down, pressed)
                PspEmuKeys.I, PspEmuKeys.J, PspEmuKeys.K, PspEmuKeys.L -> Unit // analog
                else -> {
                    logger.trace { "UnhandledKey($pressed): $keyCode" }
                }
            }

            if (pressed) {
                when (keyCode) {
                    PspEmuKeys.F7 -> {
                        go(coroutineContext) {
                            promptConfigurator.prompt()
                        }
                    }
                    PspEmuKeys.F9 -> {
                        emulator.globalTrace = !emulator.globalTrace
                    }
                    PspEmuKeys.F10 -> {
                        go(coroutineContext) { toggleHud() }
                    }
                    PspEmuKeys.F11 -> {
                        // Dump threads
                        println("THREAD_DUMP:")
                        for (thread in emulator.threadManager.threads) {
                            println("Thread[${thread.id}](${thread.name}) : ${thread.status} : ${thread.waitObject}, running = ${thread.running}, waiting = ${thread.waiting}, priority=${thread.priority}, PC=${thread.state.PC.hex}, preemptionCount=${thread.preemptionCount}, totalExecuted=${thread.state.totalExecuted}")
                        }
                    }
                }
            }

            controller.updateAnalog(
                x = when { keys[PspEmuKeys.J] -> -1f; keys[PspEmuKeys.L] -> +1f; else -> 0f; },
                y = when { keys[PspEmuKeys.I] -> -1f; keys[PspEmuKeys.K] -> +1f; else -> 0f; }
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
            y = 272.0 - 24.0 * 1
        }.onClick {
                createEmulatorWithExe(browser.openFile())
            }

        val directButton = views.simpleButton("Auto", font = hudFont).apply {
            x = 8.0
            y = 272.0 - 24.0 * 2
        }.onClick {
                agRenderer.renderMode = when (agRenderer.renderMode) {
                    AGRenderer.RenderMode.AUTO -> AGRenderer.RenderMode.NORMAL
                    AGRenderer.RenderMode.NORMAL -> AGRenderer.RenderMode.AUTO
                    else -> AGRenderer.RenderMode.AUTO
                //AGRenderer.RenderMode.AUTO -> AGRenderer.RenderMode.NORMAL
                //AGRenderer.RenderMode.NORMAL -> AGRenderer.RenderMode.DIRECT
                //AGRenderer.RenderMode.DIRECT -> AGRenderer.RenderMode.AUTO
                }

                it.view.setText(
                    when (agRenderer.renderMode) {
                        AGRenderer.RenderMode.AUTO -> "Auto"
                        AGRenderer.RenderMode.NORMAL -> "Normal"
                        AGRenderer.RenderMode.DIRECT -> "Direct"
                    }
                )
            }

        lateinit var pauseButton: View

        fun pause(set: Boolean) {
            paused = set
            pauseButton.setText(if (paused) "Resume" else "Pause")
        }

        pauseButton = views.simpleButton("Pause", font = hudFont).apply {
            x = 8.0
            y = 272.0 - 24.0 * 3
        }.onClick {
                pause(!paused)
            }!!

        val stepButton = views.simpleButton("Step", font = hudFont).apply {
            x = 8.0
            y = 272.0 - 24.0 * 4
        }.onClick {
                pause(true)
                forceSteps++
            }

        val promptButton = views.simpleButton("prompt", font = hudFont).apply {
            x = 8.0
            y = 272.0 - 24.0 * 5
        }.onClick {
                promptConfigurator.prompt()
            }


        hud += views.solidRect(96, 272, RGBA(0, 0, 0, 0xCC)).apply { enabled = false; mouseEnabled = false }
        hud += infoText
        hud += loadButton
        hud += directButton
        hud += pauseButton
        hud += stepButton
        hud += promptButton
        hud += icon0Image.apply {
            scale = 0.5
            x = 100.0
            y = 16.0
        }
        hud += titleText.apply {
            scale = 0.8
            x = 100.0
            y = 0.0
        }

        val displayView = object : View(views) {
            override fun getLocalBoundsInternal(out: Rectangle): Unit = run { out.setTo(0, 0, 480, 272) }
            override fun render(ctx: RenderContext, m: Matrix2d) {
                val startTime = Klock.currentTimeMillis()
                fpsCounter.tick(startTime.toDouble())
                //println("displayView.render.Thread: ${KorioNative.currentThreadId}")
                agRenderer.render(views, ctx, m)
                agRenderer.renderTexture(views, ctx, m)
                val endTime = Klock.currentTimeMillis()
                agRenderer.stats.renderTime = (endTime - startTime).toInt()
                infoText.text = getInfoText()
                agRenderer.stats.reset()
            }
        }

        sceneView += displayView

        debugSceneContainer = views.sceneContainer().apply {
            changeTo<DebugScene>()
        }
        sceneView += debugSceneContainer

        sceneView += hud

        //sceneView.onMove { hudOpen() }
        //sceneView.onOut { hudClose() }
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

    lateinit var debugSceneContainer: SceneContainer

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

    fun getFpsInt(): Int = getFps().roundToInt()

    fun getFps(): Double {
        if (renderCount == 0) return 0.0
        val elapsed = getSample(0) - getSample(renderCount - 1)
        return 1000.0 * (renderCount.toDouble() / elapsed)
    }
}

suspend fun AsyncStream.preload() = this.readAll().openAsync()
suspend fun AsyncStream.preloadSmall(size: Long = 4L * 1024 * 1024) = if (this.size() < size) this.preload() else this

open class LoadProcess {
    open suspend fun readIcon0(icon0: ByteArray): Unit = Unit
    open suspend fun readParamSfo(psf: Psf): Unit = Unit
}

suspend fun Emulator.loadExecutableAndStart(file: VfsFile, loadProcess: LoadProcess = LoadProcess()): PspElf {
    var stream = file.open().preloadSmall()
    var umdLikeStructure = false
    var layerName = file.basename
    var container = file.parent.jail()
    var gameName = "virtual"

    logger.warn { "Opening $file" }
    while (true) {
        if (stream.size() < 0x10) {
            logger.warn { " - Layer(${stream.size()}): Format ???" }
            invalidOp("Layer is too small")
        }
        val format = stream.detectPspFormat(layerName) ?: invalidOp("Unsupported file format '$file'")
        logger.warn { " - Layer(${stream.size()}): Format $format" }
        when (format) {
            PspFileFormat.CSO -> {
                stream = stream.openAsCso()
                layerName = "$layerName.iso"
            }
            PspFileFormat.ISO, PspFileFormat.ZIP -> {
                container = when (format) {
                    PspFileFormat.ISO -> stream.openAsIso2()
                    PspFileFormat.ZIP -> stream.openAsZip2()
                    else -> TODO("Impossible!") // @TODO: Kotlin could detect this!
                }

                var afile: VfsFile? = null
                done@ for (folder in listOf("PSP_GAME/SYSDIR", "")) {
                    umdLikeStructure = folder.isNotEmpty()
                    for (filename in listOf("EBOOT.BIN", "EBOOT.ELF", "EBOOT.PBP", "BOOT.BIN")) {
                        afile = container["$folder/$filename"]
                        if (afile.exists()) {
                            // Some BOOT.BIN files are filled with 0!
                            if (afile.readRangeBytes(0 until 4).hexString != "00000000") {
                                logger.warn { "Using $afile from iso" }
                                break@done
                            }
                        }
                    }
                }

                if (afile == null) invalidOp("Can't find any suitable executable inside $format")

                for (icon0 in listOf(container["PSP_GAME/ICON0.PNG"])) {
                    if (icon0.exists()) loadProcess.readIcon0(icon0.readAll())
                }

                for (paramSfo in listOf(container["PSP_GAME/PARAM.SFO"])) {
                    if (paramSfo.exists()) {
                        val psf = Psf(paramSfo.readAll())
                        gameName = psf.getString("DISC_ID") ?: "virtual"
                        loadProcess.readParamSfo(psf)
                    }
                }

                stream = afile.open()
            }
            PspFileFormat.PBP -> {
                val pbp = Pbp(stream)
                val icon0 = pbp.ICON0_PNG.readAll()
                if (icon0.isNotEmpty()) loadProcess.readIcon0(icon0)
                val paramSfo = pbp.PARAM_SFO.readAll()
                if (paramSfo.isNotEmpty()) {
                    val psf = Psf(pbp.PARAM_SFO)
                    gameName = psf.getString("DISC_ID") ?: "virtual"
                    loadProcess.readParamSfo(psf)
                }
                stream = pbp.PSP_DATA
            }
            PspFileFormat.ENCRYPTED_ELF -> {
                val encryptedData = stream.readAll()
                //val decryptedData = CryptedElf.decrypt(encryptedData)
                val decryptedData = CryptedElf.decrypt(encryptedData)
                if (decryptedData.detectPspFormat() != PspFileFormat.ELF) {
                    val encryptedFile = tempVfs["BOOT.BIN.encrypted"]
                    val decryptedFile = tempVfs["BOOT.BIN.decrypted"]
                    encryptedFile.writeBytes(encryptedData)
                    decryptedFile.writeBytes(decryptedData)
                    invalidOp("Error decrypting file. Written to: ${decryptedFile.absolutePath} & ${encryptedFile.absolutePath}")
                }
                //LocalVfs("c:/temp/decryptedData.bin").write(decryptedData)
                stream = decryptedData.openAsync()
            }
            PspFileFormat.ELF -> {
                when {
                    umdLikeStructure -> {
                        fileManager.currentDirectory = "umd0:"
                        fileManager.executableFile = "umd0:/PSP_GAME/USRDIR/EBOOT.BIN"
                        deviceManager.mount("game0:", container)
                        deviceManager.mount("disc0:", container)
                        deviceManager.mount("umd0:", container)
                    }
                    else -> {
                        val ngameName = PathInfo(gameName).basename
                        val PSP_GAME_folder = "PSP/GAME/$ngameName"
                        val ms_PSP_GAME_folder = "ms0:/$PSP_GAME_folder"
                        deviceManager.ms[PSP_GAME_folder].mkdirsSafe()
                        fileManager.currentDirectory = ms_PSP_GAME_folder
                        fileManager.executableFile = "$ms_PSP_GAME_folder/EBOOT.PBP"
                        deviceManager.mount(
                            ms_PSP_GAME_folder,
                            MergedVfs(
                                listOf(
                                    deviceManager.ms[PSP_GAME_folder].jail(),
                                    container
                                )
                            ).root
                        )
                    }
                }

                logger.warn { "Loading executable mounted at ARG0: ${fileManager.executableFile} : CWD: ${fileManager.currentDirectory}..." }
                for (mount in deviceManager.mountable.mounts) {
                    logger.warn { "  - Mount: ${mount.key}   -->   ${mount.value}" }
                }
                return loadElfAndSetRegisters(stream.readAll().openSync(), listOf(fileManager.executableFile))
            }
            else -> invalidOp("Unhandled format $format")
        }
    }
}

