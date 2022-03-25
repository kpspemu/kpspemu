package com.soywiz.kpspemu

import com.soywiz.korge.html.*
import com.soywiz.korge.input.*
import com.soywiz.korge.scene.*
import com.soywiz.korge.service.*
import com.soywiz.korge.view.*
import com.soywiz.korim.color.*
import com.soywiz.korim.font.*
import com.soywiz.korio.async.*
import com.soywiz.korio.lang.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.cpu.dis.*
import com.soywiz.kpspemu.hle.manager.*
import com.soywiz.kpspemu.mem.*
import com.soywiz.kpspemu.ui.*
import com.soywiz.kpspemu.util.*
import com.soywiz.kpspemu.util.expr.*
import com.soywiz.krypto.encoding.*
import kotlin.reflect.*

/**
 * You can enable this scene using F2
 * When setting values, you can use register names or mnemonic names, literals and simple expressions.
 * Example of valid expressions: pc + 0x10 + a0 + r31
 */
class DebugScene(
    val browser: Browser,
    val emulator: Emulator
) : Scene() {
    var viewAddress: Int = 0

    lateinit var registerList: GprListView
    lateinit var disassembler: DissasemblerView
    lateinit var threadList: ThreadListView
    lateinit var hexdump: HexdumpView
    lateinit var font: BitmapFont

    suspend fun askGoto() {
        val thread = emulator.threadManager.threads.firstOrNull()
        val exprStr = browser.prompt("Address", viewAddress.hex)
        val addr = evaluateIntExpr(thread?.state ?: CpuState.dummy, exprStr) and
                0b11.inv()
        println("Expr: $exprStr")
        println("Address: ${addr.hex}")
        viewAddress = addr
    }

    suspend fun askEval() {
        val thread = emulator.threadManager.threads.firstOrNull()
        val exprStr = browser.prompt("Expression", viewAddress.hex)
        val value = evaluateIntExpr(thread?.state ?: CpuState.dummy, exprStr) and
                0b11.inv()
        println("Expr: $exprStr")
        println("Result: ${value.hex}")
        browser.alert("Expr: $exprStr\nResult: ${value.hex}")
    }

    override suspend fun Container.sceneInit() {
        val sceneView = this
        sceneView.visible = false

        sceneView += SolidRect(480, 272, RGBA(0xFF, 0xFF, 0xFF, 0xAF))

        sceneView.keys {
            down {
                when (it.keyCode) {
                    PspEmuKeys.F2 -> sceneView.visible = !sceneView.visible // F2
                    else -> if (sceneView.visible) when (it.keyCode) {
                        PspEmuKeys.E -> askEval() // E
                        PspEmuKeys.G -> askGoto() // G
                        PspEmuKeys.UP -> moveUp() // UP
                        PspEmuKeys.DOWN -> moveDown() // DOWN
                        PspEmuKeys.PGUP -> moveUp(16) // PGUP
                        PspEmuKeys.PGDOWN -> moveDown(16) // PGDOWN
                        PspEmuKeys.T -> toggle() // T
                        PspEmuKeys.F3 -> toggle() // F3
                        else -> println("onKeyDown: ${it.keyCode}")
                    }
                }
            }
        }
        font = views.debugBmpFont

        sceneView += SolidRect(480, 272, Colors.BLUE.withA(0x5F)).apply {
            mouseEnabled = false
        }

        sceneView += GprListView(browser, views, font).apply {
            registerList = this
        }

        sceneView += DissasemblerView(emulator, views, font).apply {
            disassembler = this
            visible = true
            x = 96.0
            y = 8.0
        }
        sceneView += ThreadListView(emulator, views, font).apply {
            threadList = this
            visible = true
            x = 400.0
            y = 8.0
        }

        sceneView += HexdumpView(emulator, views, font).apply {
            hexdump = this
            visible = false
            x = 32.0
            y = 8.0
        }
        sceneView += views.simpleButton("[G]OTO", width = 48, height = 8, font = font).apply {
            x = 96.0
            y = 0.0
            onClick {
                launchImmediately {
                    askGoto()
                }
            }
        }
        sceneView += views.simpleButton("TOGGLE", width = 48, height = 8, font = font).apply {
            x = 140.0
            y = 0.0
            onClick {
                toggle()
            }
        }

        sceneView.addUpdater {
            if (sceneView.visible) {
                val thread = emulator.threadManager.threads.firstOrNull()
                val cpu = thread?.state ?: CpuState.dummy
                if (registerList.visible) registerList.update(cpu)
                if (disassembler.visible) disassembler.update(viewAddress, cpu.mem, cpu)
                if (threadList.visible) threadList.update()
                if (hexdump.visible) hexdump.update(viewAddress, cpu.mem, cpu)
            }
        }
    }

    val disassemblerShowing get() = disassembler.visible
    val displaceBytes: Int get() = if (disassemblerShowing) 4 else 16

    private fun moveDir(offset: Int) {
        viewAddress += offset
    }

    private fun moveUp(scale: Int = 1) = moveDir(-1 * displaceBytes * scale)
    private fun moveDown(scale: Int = 1) = moveDir(+1 * displaceBytes * scale)

    private fun toggle() {
        val v = disassembler.visible
        registerList.visible = !v
        disassembler.visible = !v
        hexdump.visible = v
    }

    class GprView(
        views: Views,
        val font: BitmapFont,
        val regName: String,
        val regSet: (value: Int) -> Unit,
        val regGet: () -> Int
    ) : Container() {
        val BG_OVER = RGBA(0, 0, 0xFF, 0xFF)
        val BG_OUT = RGBA(0, 0, 0xFF, 0x7f)
        val onGprClick = AsyncSignal<GprView>()

        val black = Colors.BLACK
        val text = textOld("", font = font).apply {
            filtering = false
            x = 0.0
            // Exception in thread "AWT-EventQueue-0" java.lang.ClassCastException: java.lang.Integer cannot be cast to com.soywiz.korim.color.RGBA
            //	at com.soywiz.kpspemu.DebugScene$GprView.<init>(DebugScene.kt:169)
            //	at com.soywiz.kpspemu.DebugScene$GprListView.<init>(DebugScene.kt:194)
            format = Html.Format(face = font, size = 8, color = black)
            autoSize = true
            bgcolor = BG_OUT
            onOver { bgcolor = BG_OVER }
            onOut { bgcolor = BG_OUT }
            onClick {
                launchImmediately(views.coroutineContext) {
                    this@GprView.onGprClick(this@GprView)
                }
            }
        }

        var value = 0
            set(value) {
                regSet(value)
                field = regGet()
                text.text = "$regName=${field.shex}"
            }

        init {
            value = regGet()
        }
    }

    class GprListView(browser: Browser, views: Views, val font: BitmapFont) : Container() {
        var state = CpuState("GprListView", GlobalCpuState(MemoryInfo.DUMMY))
        val regs = (0 until 32).map { regIndex ->
            GprView(views, font, "r$regIndex", { state.setGpr(regIndex, it) }, { state.getGpr(regIndex) }).apply {
                this@GprListView += this
                y = (regIndex * 8).toDouble()
            }
        }
        val pc = GprView(views, font, "PC", { state.setPC(it) }, { state.PC }).apply {
            this@GprListView += this
            y = (32 * 8).toDouble()
        }

        val allRegs = regs + pc

        init {
            for (reg in allRegs) {
                reg.onGprClick {
                    reg.value = evaluateIntExpr(state, browser.prompt("Set register ${reg.regName}", reg.value.hex))
                    //println("result: $result -> $result2")
                }
            }
            //addUpdatable {
            //	println(regs[0].getBounds())
            //}
        }

        fun update(state: CpuState) {
            this.state = state
            //val thread = emulator.threadManager.threads.first()
            //thread.state.getGpr()
            for (n in 0 until 32) {
                regs[n].value = state.getGpr(n)
            }
            pc.value = state.PC
        }
    }

    class HexdumpLineView(val emulator: Emulator, views: Views, val lineNumber: Int, val font: BitmapFont) :
        Container() {
        val black = Colors.BLACK
        val text = textOld("-", font = font).apply {
            filtering = false
            x = 0.0
            y = (lineNumber * 8).toDouble()
            autoSize = true
            // Exception in thread "AWT-EventQueue-0" java.lang.ClassCastException: java.lang.Integer cannot be cast to com.soywiz.korim.color.RGBA
            //	at com.soywiz.kpspemu.DebugScene$HexdumpLineView.<init>(DebugScene.kt:240)
            //	at com.soywiz.kpspemu.DebugScene$HexdumpView.<init>(DebugScene.kt:258)
            //format = Html.Format(face = Html.FontFace.Bitmap(font), size = 8, color = Colors.BLACK)
            format = Html.Format(face = font, size = 8, color = black)
        }

        fun update(addr: Int, mem: Memory, state: CpuState) {
            var line = ""
            line += addr.shex
            line += " | "
            for (n in 0 until 16) line += "%02X".format(mem.lbuSafe(addr + n))
            line += " | "
            for (n in 0 until 16) {
                val c = mem.lbuSafe(addr + n)
                line += if (c < 32) "." else "" + c.toChar()
            }
            text.text = line
        }
    }

    class HexdumpView(val emulator: Emulator, views: Views, val font: BitmapFont) : Container() {
        val texts = (0 until 32).map { HexdumpLineView(emulator, views, it, font) }

        init {
            for (text in texts) {
                this += text
            }
        }

        fun update(startAddress: Int, memory: Memory, state: CpuState) {
            for ((n, text) in texts.withIndex()) {
                val address = startAddress + n * 16
                text.update(address, memory, state)
            }
        }
    }

    class DissasemblerLineView(val emulator: Emulator, views: Views, val lineNumber: Int, val font: BitmapFont) :
        Container() {
        val BG_NORMAL = RGBA(0xFF, 0xFF, 0xFF, 0x99)
        val BG_PC = RGBA(0, 0, 0xFF, 0x99)
        val BG_BREAKPOINT = RGBA(0xFF, 0, 0, 0x99)

        var addr = 0
        val onLineClick = AsyncSignal<Unit>()
        var over = false

        val black = Colors.BLACK
        val text = textOld("-", font = font).apply {
            this@DissasemblerLineView += this
            filtering = false
            x = 0.0
            y = (lineNumber * 8).toDouble()
            autoSize = true
            //Exception in thread "AWT-EventQueue-0" java.lang.ClassCastException: java.lang.Integer cannot be cast to com.soywiz.korim.color.RGBA
            //	at com.soywiz.kpspemu.DebugScene$DissasemblerLineView.<init>(DebugScene.kt:290)
            //	at com.soywiz.kpspemu.DebugScene$DissasemblerView.<init>(DebugScene.kt:322)
            //format = Html.Format(face = Html.FontFace.Bitmap(font), size = 8, color = Colors.BLACK)
            format = Html.Format(face = font, size = 8, color = black)
            onOver { over = true }
            onOut { over = false }
            onClick {
                launchImmediately(emulator.coroutineContext) {
                    onLineClick(Unit)
                }
            }
        }

        fun update(addr: Int, memory: Memory, state: CpuState, emulator: Emulator) {
            this.addr = addr
            val atPC = addr == state.PC
            text.bgcolor = when {
            //addr == state.PC -> BG_PC
                emulator.breakpoints[addr] -> BG_BREAKPOINT
                else -> BG_NORMAL
            }
            if (over) {
                text.bgcolor = text.bgcolor.withA(0xFF)
            }

            val prefix = if (atPC) "*" else " "

            text.text = "$prefix${addr.shex}:" + try {
                Disassembler.disasm(addr, memory.lwSafe(addr), emulator.nameProvider)
            } catch (e: Throwable) {
                //e.printStackTrace()
                "ERROR"
            } + " "
        }
    }

    class DissasemblerView(val emulator: Emulator, views: Views, val font: BitmapFont) : Container() {
        val texts = (0 until 32).map { DissasemblerLineView(emulator, views, it, font) }

        init {
            for (text in texts) {
                this += text
                text.onLineClick {
                    emulator.breakpoints.toggle(text.addr)
                    println("Toggle breakpoint at: ${text.addr.hex}")
                }
            }
        }

        fun update(startAddress: Int, memory: Memory, state: CpuState) {
            for ((n, text) in texts.withIndex()) {
                val address = startAddress + n * 4
                text.update(address, memory, state, emulator)
            }
        }
    }

    class ThreadView(val emulator: Emulator, views: Views, val font: BitmapFont) : Container() {
        val BG_NORMAL = RGBA(0xFF, 0xFF, 0xFF, 0x99)
        val BG_PC = RGBA(0, 0, 0xFF, 0x99)
        val BG_BREAKPOINT = RGBA(0xFF, 0, 0, 0x99)

        var lineId = 0
        var lineThread: PspThread? = null
        var selected = false
        var addr = 0
        val onLineClick = AsyncSignal<Unit>()
        var over = false

        val text = TextOld().apply {
            autoSize = true
            format = Html.Format(face = font, size = 8, color = Colors.BLACK)
            filtering = false
            onOver { over = true }
            onOut { over = false }
            onClick {
                launchImmediately(emulator.coroutineContext) {
                    onLineClick(Unit)
                }
            }
            this@ThreadView += this
        }

        fun update() {
            text.bgcolor = if (selected) BG_BREAKPOINT else BG_NORMAL
            if (over) {
                text.bgcolor = text.bgcolor.withA(0xFF)
            }
        }
    }

    class ThreadListView(val emulator: Emulator, views: Views, val font: BitmapFont) : Container() {
        val viewPool = AutoArrayList {
            ThreadView(emulator, views, font).apply {
                onLineClick {
                    clickedLine(this)
                }
            }
        }

        fun clickedLine(threadView: ThreadView) {
            val tracing = !(threadView.lineThread?.tracing ?: false)
            println("Tracing for thread: ${threadView.lineThread?.name} = $tracing")
            threadView.lineThread?.tracing = tracing
        }

        fun update() {
            removeChildren()
            for ((index, thread) in emulator.threadManager.threads.sortedBy { it.name }.withIndex()) {
                val line = viewPool[index]
                line.lineId = index
                line.lineThread = thread
                addChild(line)
                line.y = (index * 8).toDouble()
                line.text.text = thread.name
                line.selected = thread.tracing
                line.update()
            }
        }
    }
}

private fun evaluateExpr(state: CpuState, str: String): Any? {
    return ExprNode.parse(str).eval(object : ExprNode.EvalContext() {
        override fun getProp(name: String): KMutableProperty0<*>? {
            val out = getProp2(name)
            println("getProp: $name -> $out")
            return out
        }

        fun getProp2(name: String): KMutableProperty0<*>? {
            val rname = name.toLowerCase()
            if (rname.startsWith("r")) {
                return rname.substr(1).toIntOrNull()?.let { state.getGprProp(it) }
            }
            return when (rname) {
                "pc" -> state.getPCRef()
                else -> {
                    CpuState.gprInfosByMnemonic[rname]?.let { state.getGprProp(it.index) }
                }
            }
        }
    })
}

private fun evaluateIntExpr(state: CpuState, str: String) = evaluateExpr(state, str).toDynamicInt()
