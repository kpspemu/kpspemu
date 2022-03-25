package com.soywiz.kpspemu

import com.soywiz.compat.*
import com.soywiz.klock.*
import com.soywiz.klogger.*
import com.soywiz.kmem.*
import com.soywiz.korge.*
import com.soywiz.korge.bitmapfont.*
import com.soywiz.korge.html.*
import com.soywiz.korge.input.*
import com.soywiz.korge.render.*
import com.soywiz.korge.scene.*
import com.soywiz.korge.service.*
import com.soywiz.korge.time.*
import com.soywiz.korge.tween.*
import com.soywiz.korge.view.*
import com.soywiz.korge.view.Container
import com.soywiz.korge.view.Image
import com.soywiz.korim.bitmap.*
import com.soywiz.korim.color.*
import com.soywiz.korim.font.*
import com.soywiz.korim.format.*
import com.soywiz.korim.vector.*
import com.soywiz.korinject.*
import com.soywiz.korio.*
import com.soywiz.korio.async.*
import com.soywiz.korio.error.*
import com.soywiz.korio.file.*
import com.soywiz.korio.file.std.*
import com.soywiz.korio.lang.*
import com.soywiz.korio.stream.*
import com.soywiz.korio.util.*
import com.soywiz.korma.*
import com.soywiz.korma.geom.*
import com.soywiz.korma.geom.Rectangle
import com.soywiz.kpspemu.ctrl.*
import com.soywiz.kpspemu.display.*
import com.soywiz.kpspemu.format.*
import com.soywiz.kpspemu.format.elf.*
import com.soywiz.kpspemu.ge.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.native.*
import com.soywiz.kpspemu.ui.*
import com.soywiz.kpspemu.util.*
import com.soywiz.kpspemu.util.io.*
import kotlinx.coroutines.*
import kotlin.coroutines.*
import kotlin.math.*
import kotlin.reflect.*
import com.soywiz.kmem.umod
import com.soywiz.korev.*
import com.soywiz.korgw.*
import com.soywiz.korim.text.*
import com.soywiz.krypto.encoding.*
import kotlinx.coroutines.channels.*
import kotlinx.coroutines.flow.*
import kotlin.jvm.*
import com.soywiz.korio.error.invalidOp as invalidOp1

suspend fun main(args: Array<String>) = Main.main(args)

suspend fun main() = Main.main(arrayOf())

object Main {
    @JvmStatic
    suspend fun main(args: Array<String>) = Korge(Korge.Config(KpspemuModule))
}

object KpspemuModule : Module() {
    //override val clearEachFrame: Boolean = false

    override val clearEachFrame: Boolean = true
    override val mainScene: KClass<out Scene> = KpspemuMainScene::class
    //override val mainScene: KClass<out Scene> = EmptyScene::class
    override val title: String = "kpspemu - ${Kpspemu.VERSION}"
    //override val iconImage: SizedDrawable? = com.soywiz.korim.vector.format.SVG(KpspemuAssets.LOGO)
    override val size: SizeInt get() = SizeInt(480, 272)
    override val windowSize: SizeInt get() = SizeInt(480 * 2, 272 * 2)
    //override val targetFps = PspDisplay.VERTICAL_SYNC_HZ // @TODO: Bad performance!

    override suspend fun AsyncInjector.configure() {
        this.mapSingleton { Emulator(get<GameWindow>().coroutineDispatcher) }
            .mapSingleton { PromptConfigurator(get(), get(), get()) }
            .mapPrototype { KpspemuMainScene(get(), get(), get(), get()) }
            .mapPrototype { DebugScene(get(), get()) }
            .mapPrototype { TouchButtonsScene(get()) }
            .mapSingleton { Browser(get()) }

    }
}

abstract class SceneWithProcess : Scene() {
    private val processes = arrayListOf<Deferred<Unit>>()

    fun registerSceneProcess(callback: suspend () -> Unit) {
        processes += asyncImmediately(coroutineContext) { callback() }
    }

    override suspend fun sceneDestroy() {
        super.sceneDestroy()
        //for (process in processes) process.cancel(DestroyException)
        for (process in processes) process.cancel()
    }
}

class KpspemuMainScene(
    val browser: Browser,
    override val emulator: Emulator,
    val args: Korge.ModuleArgs,
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
    lateinit var hudFont: BitmapFont
    var running = true
    var ended = false
    var paused = false
    var forceSteps = 0

    val buttonMapping = mapOf(
        GameButton.BUTTON0 to PspCtrlButtons.cross,
        GameButton.BUTTON1 to PspCtrlButtons.circle,
        GameButton.BUTTON2 to PspCtrlButtons.square,
        GameButton.BUTTON3 to PspCtrlButtons.triangle,

        GameButton.L1 to PspCtrlButtons.leftTrigger,
        GameButton.R1 to PspCtrlButtons.rightTrigger,

        GameButton.SELECT to PspCtrlButtons.select,
        GameButton.START to PspCtrlButtons.start,
        GameButton.SYSTEM to PspCtrlButtons.home,

        GameButton.L2 to PspCtrlButtons.select,
        GameButton.R2 to PspCtrlButtons.start,

        GameButton.L3 to PspCtrlButtons.select,
        GameButton.R3 to PspCtrlButtons.start,

        GameButton.LEFT to PspCtrlButtons.left,
        GameButton.RIGHT to PspCtrlButtons.right,
        GameButton.UP to PspCtrlButtons.up,
        GameButton.DOWN to PspCtrlButtons.down
    )

    val gamepadButtons = GameButton.values()
    val gamepadButtonsStatus = DoubleArray(GameButton.MAX) { Double.NaN }

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

    val icon0Image: Image by lazy {
        Image(Bitmaps.transparent).apply {
            alpha = 0.5
        }
    }
    val titleText by lazy {
        Text("", font = hudFont).apply {
            alpha = 0.5
        }
    }

    fun setIcon0Bitmap(bmp: Bitmap) {
        icon0Image.bitmap = bmp.slice()
    }

    suspend fun createEmulatorWithExe(exeFile: VfsFile) {
        //println(exeFile.open().slice(0 until 8).readAll().hex)
        //println(exeFile.open().slice(8 until 16).readAll().hex)

        setIcon0Bitmap(Bitmap32(144, 80))
        titleText.text = ""

        resetEmulator()
        emulator.registerNativeModules()
        emulator.loadExecutableAndStart(exeFile, object : LoadProcess() {
            override suspend fun readIcon0(icon0: ByteArray) {
                //icon0.writeToFile("c:/temp/icon0.png")
                try {
                    setIcon0Bitmap(PNG.read(icon0, "file.png").toBMP32())
                } catch (e: Throwable) {
                    println("ERROR at loadExecutableAndStart")
                    e.printStackTrace()
                }
            }

            override suspend fun readParamSfo(psf: Psf) {
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
                    val startTime = Klock.currentTimeMillisDouble()
                    try {
                        emulator.threadManager.step()
                    } catch (e: Throwable) {
                        println("ERROR at cpuProcess")
                        e.printStackTrace()
                        running = false
                    }
                    val endTime = Klock.currentTimeMillisDouble()
                    agRenderer.stats.cpuTime = (endTime - startTime).toInt()
                } else {
                    if (!ended) {
                        ended = true
                        println("COMPLETED[1]")
                        display.crash()
                        println("COMPLETED[2]")
                    }
                }
                emulator.threadManager.waitThreadChange()
            } else {
                delay(MS_10)
            }
        }
    }

    suspend fun displayProcess() {
        var accumulatedMs = 0.0
        while (true) {
            val toWait = PspDisplay.VERTICAL_MS + accumulatedMs
            val toWaitInt = toWait.toInt()
            accumulatedMs = toWait - toWaitInt
            controller.startFrame(timeManager.getTimeInMicrosecondsInt())
            emulator.interruptManager.dispatchVsync()
            sceneView.delay(toWaitInt.milliseconds)
            controller.endFrame()
        }
    }

    private suspend fun loadMainFont() {
        try {
            hudFont = resourcesRoot["lucida_console32.fnt"].readBitmapFont()
            //hudFont = getDebugBmpFontOnce()
        } catch (e: Throwable) {
            //e.printStackTrace()
            hudFont = views.debugBmpFont
        }
    }

    private fun requestUserLoadFile() {
        launchImmediately(coroutineContext) {
            val file = views.gameWindow.openFileDialog().firstOrNull()
            //val file = browser.openFile()
            launchImmediately { hudClose() }
            if (file != null) {
                createEmulatorWithExe(file)
            }
        }
    }

    val keys = LinkedHashMap<Key, Boolean>()

    fun keyPressed(key: Key): Boolean = keys.getOrElse(key) { false }

    private fun updateKey(key: Key, pressed: Boolean) {
        //println("updateKey: $key, $pressed")
        keys[key] = pressed
        when (key) {
            Key.ENTER -> controller.updateButton(PspCtrlButtons.start, pressed)
            Key.SPACE -> controller.updateButton(PspCtrlButtons.select, pressed)
            Key.W -> controller.updateButton(PspCtrlButtons.triangle, pressed)
            Key.A -> controller.updateButton(PspCtrlButtons.square, pressed)
            Key.S -> controller.updateButton(PspCtrlButtons.cross, pressed)
            Key.D -> controller.updateButton(PspCtrlButtons.circle, pressed)
            Key.Q -> controller.updateButton(PspCtrlButtons.leftTrigger, pressed)
            Key.E -> controller.updateButton(PspCtrlButtons.rightTrigger, pressed)
            Key.LEFT -> controller.updateButton(PspCtrlButtons.left, pressed)
            Key.UP -> controller.updateButton(PspCtrlButtons.up, pressed)
            Key.RIGHT -> controller.updateButton(PspCtrlButtons.right, pressed)
            Key.DOWN -> controller.updateButton(PspCtrlButtons.down, pressed)
            Key.I, Key.J, Key.K, Key.L -> Unit // analog
            else -> {
                logger.trace { "UnhandledKey($pressed): $key" }
            }
        }

        if (pressed) {
            when (key) {
                Key.F7 -> {
                    launchImmediately(coroutineContext) {
                        promptConfigurator.prompt()
                    }
                }
                Key.F9 -> {
                    emulator.globalTrace = !emulator.globalTrace
                }
                Key.F10 -> {
                    launchImmediately(coroutineContext) { hudToggle() }
                }
                Key.F11 -> {
                    // Dump threads
                    println("THREAD_DUMP:")
                    for (thread in emulator.threadManager.threads) {
                        println("Thread[${thread.id}](${thread.name}) : ${thread.status} : ${thread.waitObject}, running = ${thread.running}, waiting = ${thread.waiting}, priority=${thread.priority}, PC=${thread.state.PC.hex}, preemptionCount=${thread.preemptionCount}, totalExecuted=${thread.state.totalExecuted}")
                    }
                }
            }
        }

        controller.updateAnalog(
            x = when { keyPressed(Key.J) -> -1f; keyPressed(Key.L) -> +1f; else -> 0f; },
            y = when { keyPressed(Key.I) -> -1f; keyPressed(Key.K) -> +1f; else -> 0f; }
        )
    }

    override suspend fun Container.sceneInit() {
        val sceneView = this
        registerSceneProcess { cpuProcess() }
        registerSceneProcess { displayProcess() }
        //val func = function(DClass(CpuState::class), DVOID) {
        //	SET(p0[CpuState::_PC], 7.lit)
        //}.generateDynarek()
        //
        //val cpuState = CpuState(GlobalCpuState(), Memory())
        //val result = func(cpuState)
        //println("PC: ${cpuState.PC.hex}")

        println("RESET EMULATOR[0]")
        resetEmulator()
        println("RESET EMULATOR[1]")
        println("KPSPEMU: ${Kpspemu.VERSION}")

        println("Loading main font...")
        loadMainFont()
        println("Loaded main font")

        hud = Container()
        //createEmulatorWithExe(exeFile)


        //hud.alpha = 0.0
        //hud.mouseEnabled = false

        fun getInfoText(): String {
            val out = arrayListOf<String>()
            out += "kpspemu-${OS.rawName}"
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

        val infoText = Text(getInfoText(), textSize = 8.0, font = hudFont).apply {
            x = 4.0
            y = 4.0
        }

        emulator.onLoadPress {
            requestUserLoadFile()
        }

        val loadButton = views.simpleButton("Load...", font = hudFont).apply {
            x = 8.0
            y = 272.0 - 20.0 * 1
        }.onClick {
            requestUserLoadFile()
        }

        val directButton = views.simpleButton("Auto", font = hudFont).apply {
            x = 8.0
            y = 272.0 - 20.0 * 2
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
            y = 272.0 - 20.0 * 3
        }.onClick {
            pause(!paused)
        }!!

        val stepButton = views.simpleButton("Step", font = hudFont).apply {
            x = 8.0
            y = 272.0 - 20.0 * 4
        }.onClick {
            pause(true)
            forceSteps++
        }

        val promptButton = views.simpleButton("Prompt", font = hudFont).apply {
            x = 8.0
            y = 272.0 - 20.0 * 5
        }.onClick {
            launchImmediately {
                promptConfigurator.prompt()
            }
        }

        val interpretedButton = views.simpleButton("", font = hudFont).apply {
            x = 8.0
            y = 272.0 - 20.0 * 6

            fun update() {
                this["label"].setText(if (emulator.interpreted) "Interpreted" else "Dynarek")
            }

            onClick {
                emulator.interpreted = !emulator.interpreted
                update()
            }
            update()
        }


        hud += SolidRect(96, 272, RGBA(0, 0, 0, 0xCC)).apply { enabled = false; mouseEnabled = false }
        hud += infoText
        hud += loadButton
        hud += directButton
        hud += pauseButton
        hud += stepButton
        hud += promptButton
        hud += interpretedButton
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

        val displayView = object : View() {
            override fun getLocalBoundsInternal(out: Rectangle): Unit = run { out.setTo(0, 0, 480, 272) }
            override fun renderInternal(ctx: RenderContext) {
                val m = globalMatrix
                val startTime = Klock.currentTimeMillisDouble()
                fpsCounter.tick(startTime)
                //println("displayView.render.Thread: ${KorioNative.currentThreadId}")
                agRenderer.render(views, ctx, m)
                agRenderer.renderTexture(views, ctx, m)
                val endTime = Klock.currentTimeMillisDouble()
                agRenderer.stats.renderTime = (endTime - startTime).toInt()
                infoText.text = getInfoText()
                agRenderer.stats.reset()
            }
        }

        sceneView += displayView

        debugSceneContainer = SceneContainer(views).apply {
            changeTo<DebugScene>()
        }
        sceneView += debugSceneContainer

        val dropContainer = Container().apply {
            visible = false
            this += SolidRect(1024, 1024, RGBA(0xA0, 0xA0, 0xA0, 0x7F))
            this += TextOld("Drop ZIP, ISO, PBP or ELF files here...", font = hudFont).apply {
                //format = Html.Format(format, align = Html.Alignment.MIDDLE_CENTER)
                format = Html.Format(format, align = TextAlignment.LEFT)
                x = 480.0 * 0.5
                y = 272.0 * 0.5
            }
        }
        sceneView += dropContainer

        sceneView += SceneContainer(views).apply { changeTo<TouchButtonsScene>() }
        sceneView += hud
        hud.alpha = 0.0
        hud.mouseEnabled = false

        //sceneView.onMove { hudOpen() }
        //sceneView.onOut { hudClose() }
        //sceneView.onClick { toggleHud() }
        emulator.onHomePress {
            println("emulator.onHomePress")
            asyncImmediately(coroutineContext) {
                println("emulator.onHomePress.async")
                hudToggle()
            }
        }
        //sceneView.onKeyTyped { println(it.keyCode) }

        sceneView.keys {
            down {
                updateKey(it.key, true)
            }
            up {
                updateKey(it.key, false)
            }
        }

        // Not available anymore
        /*
        cancellables += injector.getOrNull(Frame::class)?.onDropFiles(
            enter = {
                dropContainer.visible = true
                println("DROP ENTER")
                true
            },
            exit = {
                dropContainer.visible = false
                println("DROP EXIT")
            },
            drop = { files ->
                dropContainer.visible = false
                println("DROPFILES: $files")
                launchImmediately(coroutineContext) {
                    sleep(0.1.seconds)
                    createEmulatorWithExe(files.first())
                }
            }
        ) ?: Closeable { }
        */

        KPspEmuNative.initialization()
        if (OS.isBrowserJs) {
            val hash = KPspEmuNative.documentLocationHash.trim('#')
            val location = hash.trim('#').trim()
            println("Hash:$hash, Location:$location")
            if (location.isNotEmpty()) {
                hudCloseImmediate()
                createEmulatorWithExe(localCurrentDirVfs[location])
            }
        }

        sceneView.gamepad {
            button { playerId, pressed, button, value ->
                val gamepad = playerId
                if (gamepadButtonsStatus[button.index] != value) {
                    gamepadButtonsStatus[button.index] = value
                    val pspButton = buttonMapping[button]
                    if (pspButton != null) {
                        emulator.controller.updateButton(pspButton, value >= 0.5)
                    }
                    when (button) {
                        GameButton.LX -> emulator.controller.currentFrame.lx = (((value + 1.0) / 2.0) * 255.0).toInt()
                        GameButton.LY -> emulator.controller.currentFrame.ly = (((-value + 1.0) / 2.0) * 255.0).toInt()
                    }
                }
            }
        }

        if (args.args.isNotEmpty()) {
            launchAsap {
                createEmulatorWithExe(resourcesVfs[args.args.first()])
            }
        }
        //sceneView.addEventListener<StageResizedEvent> { e ->
        //    println("RESIZED!")
        //}
    }

    lateinit var debugSceneContainer: SceneContainer

    val fpsCounter = FpsCounter()
    val hudQueue = AsyncThread()

    suspend fun hudToggle(open: Boolean = hud.alpha < 0.5) {
        println("hudToggle: $open")
        if (open) {
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

class EmulatorLoader(val emulator: Emulator, var file: VfsFile, val loadProcess: LoadProcess = LoadProcess()) {
    lateinit var stream: AsyncStream
    var umdLikeStructure = false
    var layerName = file.basename
    var container = file.parent.jail()
    var gameName = "virtual"
    var steps = 0
    val logger = emulator.logger
    var format: PspFileFormat? = null

    suspend fun loadCso() {
        stream = stream.openAsCso()
        layerName = "$layerName.iso"
    }

    suspend fun loadElf(): PspElf {
        when {
            umdLikeStructure -> {
                emulator.fileManager.currentDirectory = "umd0:"
                emulator.fileManager.executableFile = "umd0:/PSP_GAME/USRDIR/EBOOT.BIN"
                emulator.deviceManager.mount("game0:", container)
                emulator.deviceManager.mount("disc0:", container)
                emulator.deviceManager.mount("umd0:", container)
            }
            else -> {
                val ngameName = PathInfo(gameName).baseName
                val PSP_GAME_folder = "PSP/GAME/$ngameName"
                val ms_PSP_GAME_folder = "ms0:/$PSP_GAME_folder"
                emulator.deviceManager.ms[PSP_GAME_folder].mkdirsSafe()
                emulator.fileManager.currentDirectory = ms_PSP_GAME_folder
                emulator.fileManager.executableFile = "$ms_PSP_GAME_folder/EBOOT.PBP"
                emulator.deviceManager.mount(
                    ms_PSP_GAME_folder,
                    MergedVfs(
                        listOf(
                            emulator.deviceManager.ms[PSP_GAME_folder].jail(),
                            container
                        )
                    ).root
                )
            }
        }

        logger.warn { "Loading executable mounted at ARG0: ${emulator.fileManager.executableFile} : CWD: ${emulator.fileManager.currentDirectory}..." }
        for (mount in emulator.deviceManager.mountable.mounts) {
            logger.warn { "  - Mount: ${mount.key}   -->   ${mount.value}" }
        }
        return emulator.loadElfAndSetRegisters(stream.readAll().openSync(), listOf(emulator.fileManager.executableFile))
    }

    suspend fun loadEncryptedElf() {
        val encryptedData = stream.readAll()
        //val decryptedData = CryptedElf.decrypt(encryptedData)
        val decryptedData = CryptedElf.decrypt(encryptedData)
        if (decryptedData.detectPspFormat() != PspFileFormat.ELF) {
            val encryptedFile = tempVfs["BOOT.BIN.encrypted"]
            val decryptedFile = tempVfs["BOOT.BIN.decrypted"]
            encryptedFile.writeBytes(encryptedData)
            decryptedFile.writeBytes(decryptedData)
            invalidOp1("Error decrypting file. Written to: ${decryptedFile.absolutePath} & ${encryptedFile.absolutePath}")
        }
        //LocalVfs("c:/temp/decryptedData.bin").write(decryptedData)
        stream = decryptedData.openAsync()
    }

    suspend fun loadPbp() {
        val pbp = Pbp(stream)
        val icon0 = pbp.ICON0_PNG.readAll()
        println("icon0.size:${icon0.size}")
        if (icon0.isNotEmpty()) loadProcess.readIcon0(icon0)
        val paramSfo = pbp.PARAM_SFO.readAll()
        println("paramSfo.size:${paramSfo.size}:${pbp.PARAM_SFO.size()}")
        if (paramSfo.isNotEmpty()) {
            val psf = Psf(pbp.PARAM_SFO)
            gameName = psf.getString("DISC_ID") ?: "virtual"
            loadProcess.readParamSfo(psf)
        }
        stream = pbp.PSP_DATA
    }

    suspend fun loadIsoOrZip() {
        logger.warn { " - Reading file (ISO/ZIP)" }

        container = when (format) {
            PspFileFormat.ISO -> stream.openAsPspIso()
            PspFileFormat.ZIP -> stream.openAsZip()
            else -> TODO("Impossible!") // @TODO: Kotlin could detect this!
        }

        logger.warn { " - Searching for executable" }

        val ebootNames = linkedSetOf("EBOOT.BIN", "EBOOT.ELF", "EBOOT.PBP", "BOOT.BIN")
        var afile: VfsFile? = null
        done@ for (folder in listOf("PSP_GAME/SYSDIR", "")) {
            umdLikeStructure = folder.isNotEmpty()
            for (filename in ebootNames) {
                val tfile = container["$folder/$filename"]
                if (tfile.exists()) {
                    // Some BOOT.BIN files are filled with 0!
                    if (tfile.readRangeBytes(0 until 4).hex != "00000000") {
                        afile = tfile
                        logger.warn { "Using $afile from iso" }
                        break@done
                    }
                }
            }
        }

        // Search EBOOT.PBP in folders (2 levels max) inside zip files
        if (format == PspFileFormat.ZIP && afile == null) {
            logger.warn { " - Searching for executable in subfolders" }

            val suitableFiles = container
                .listRecursive { it.fullname.count { it == '/' } <= 2 }
                .toList()
                .filter { it.basename.toUpperCase() in ebootNames }
                .sortedBy { it.fullname.count { it == '/' } }
            afile = suitableFiles.firstOrNull()
            // Rebase container
            if (afile != null) {
                container = afile.parent.jail()
                //println(afile.parent)
            }
        }

        if (afile == null) invalidOp1("Can't find any suitable executable inside $format")

        for (icon0 in listOf(container["PSP_GAME/ICON0.PNG"])) {
            if (icon0.exists()) {
                println("icon0.exists:${icon0.exists()}")
                val icon0Bytes = icon0.readAll()
                println("icon0Bytes:${icon0Bytes.size}")
                loadProcess.readIcon0(icon0Bytes)
            }
        }

        for (paramSfo in listOf(container["PSP_GAME/PARAM.SFO"])) {
            if (paramSfo.exists()) {
                println("container.listNames():" + container.listNames())
                println("paramSfo.stat:${paramSfo.stat()}")
                println("paramSfo.exists:${paramSfo.exists()}")
                val psfBytes = paramSfo.readAll()
                println("paramSfo:${psfBytes.size}")
                val psf = Psf(psfBytes)
                gameName = psf.getString("DISC_ID") ?: "virtual"
                loadProcess.readParamSfo(psf)
            }
        }

        stream = afile!!.open()
    }

    suspend fun loadExecutableAndStart(): PspElf {
        //stream = file.open().preloadSmall().buffered()
        stream = file.open().preloadSmall()

        umdLikeStructure = false
        layerName = file.basename
        container = file.parent.jail()
        gameName = "virtual"
        steps = 0

        logger.warn { "Opening $file" }
        while (true) {
            if (stream.size() < 0x10) {
                logger.warn { " - Layer(${stream.size()}): Format ???" }
                invalidOp1("Layer is too small")
            }
            format = stream.detectPspFormat(layerName) ?: invalidOp1("Unsupported file format '$file'")
            steps++
            if (steps >= 10) {
                invalidOp1("Too much containers!")
            }
            logger.warn { " - Layer(${stream.size()}): Format $format" }
            when (format) {
                PspFileFormat.CSO -> loadCso()
                PspFileFormat.ISO -> loadIsoOrZip()
                PspFileFormat.ZIP -> loadIsoOrZip()
                PspFileFormat.PBP -> loadPbp()
                PspFileFormat.ENCRYPTED_ELF -> loadEncryptedElf()
                PspFileFormat.ELF -> return loadElf()
            }
        }
    }
}

suspend fun Emulator.loadExecutableAndStart(file: VfsFile, loadProcess: LoadProcess = LoadProcess()): PspElf {
    return EmulatorLoader(this, file, loadProcess).loadExecutableAndStart()
}

// @TODO: @JS @BUG: THIS FUNCTION HAS PROBLEMS WITH Kotlin.JS AND ENTERS IN AN INFINITE LOOP, SO I HAVE SPLITTED IT
/*
suspend fun Emulator.loadExecutableAndStart(file: VfsFile, loadProcess: LoadProcess = LoadProcess()): PspElf {
    var stream = file.open().preloadSmall()
    var umdLikeStructure = false
    var layerName = file.basename
    var container = file.parent.jail()
    var gameName = "virtual"
    var steps = 0

    logger.warn { "Opening $file" }
    while (true) {
        if (stream.size() < 0x10) {
            logger.warn { " - Layer(${stream.size()}): Format ???" }
            invalidOp("Layer is too small")
        }
        val format = stream.detectPspFormat(layerName) ?: invalidOp("Unsupported file format '$file'")
        steps++
        if (steps >= 10) {
            invalidOp("Too much containers!")
        }
        logger.warn { " - Layer(${stream.size()}): Format $format" }
        when (format) {
            PspFileFormat.CSO -> {
                stream = stream.openAsCso()
                layerName = "$layerName.iso"
            }
            PspFileFormat.ISO, PspFileFormat.ZIP -> {
                logger.warn { " - Reading file (ISO/ZIP)" }

                container = when (format) {
                    PspFileFormat.ISO -> stream.openAsIso2()
                    PspFileFormat.ZIP -> stream.openAsZip2()
                    else -> TODO("Impossible!") // @TODO: Kotlin could detect this!
                }

                logger.warn { " - Searching for executable" }

                val ebootNames = linkedSetOf("EBOOT.BIN", "EBOOT.ELF", "EBOOT.PBP", "BOOT.BIN")
                var afile: VfsFile? = null
                done@ for (folder in listOf("PSP_GAME/SYSDIR", "")) {
                    umdLikeStructure = folder.isNotEmpty()
                    for (filename in ebootNames) {
                        val tfile = container["$folder/$filename"]
                        if (tfile.exists()) {
                            // Some BOOT.BIN files are filled with 0!
                            if (tfile.readRangeBytes(0 until 4).hexString != "00000000") {
                                afile = tfile
                                logger.warn { "Using $afile from iso" }
                                break@done
                            }
                        }
                    }
                }

                // Search EBOOT.PBP in folders (2 levels max) inside zip files
                //if (format == PspFileFormat.ZIP && afile == null) {
                //    logger.warn { " - Searching for executable in subfolders" }
                //
                //    val suitableFiles = container
                //        .listRecursive { it.fullname.count { it == '/' } <= 2 }
                //        .toList()
                //        .filter { it.basename.toUpperCase() in ebootNames }
                //        .sortedBy { it.fullname.count { it == '/' } }
                //    afile = suitableFiles.firstOrNull()
                //    // Rebase container
                //    if (afile != null) {
                //        container = afile.parent.jail()
                //        //println(afile.parent)
                //    }
                //}

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
*/