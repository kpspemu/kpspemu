package com.soywiz.kpspemu.hle.modules


import com.soywiz.kmem.*
import com.soywiz.korio.lang.*
import com.soywiz.korio.stream.*
import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*
import com.soywiz.kpspemu.hle.manager.*


@Suppress("UNUSED_PARAMETER")
class StdioForUser(emulator: Emulator) :
    SceModule(emulator, "StdioForUser", 0x40010011, "iofilemgr.prx", "sceIOFileManager") {
    val fileDescriptors get() = emulator.fileManager.fileDescriptors

    class StdioStream(val id: Int) : AsyncStreamBase() {
        suspend override fun close() = Unit
        suspend override fun getLength(): Long = 0L
        suspend override fun setLength(value: Long) = Unit

        suspend override fun read(position: Long, buffer: ByteArray, offset: Int, len: Int): Int {
            TODO("Not implemented StdioStream.read")
        }

        suspend override fun write(position: Long, buffer: ByteArray, offset: Int, len: Int) {
            print(buffer.readByteArray(offset, len).toString(UTF8))
        }
    }

    private var stdin: FileDescriptor =
        fileDescriptors.alloc().apply { stream = StdioStream(1).toAsyncStream(); file = stream.asVfsFile("/dev/stdin") }
    private var stdout: FileDescriptor = fileDescriptors.alloc()
        .apply { stream = StdioStream(2).toAsyncStream(); file = stream.asVfsFile("/dev/stdout") }
    private var stderr: FileDescriptor = fileDescriptors.alloc()
        .apply { stream = StdioStream(3).toAsyncStream(); file = stream.asVfsFile("/dev/stderr") }

    fun sceKernelStdin(): Int = stdin.id
    fun sceKernelStdout(): Int = stdout.id
    fun sceKernelStderr(): Int = stderr.id

    fun sceKernelStdioLseek(cpu: CpuState): Unit = UNIMPLEMENTED(0x0CBB0571)
    fun sceKernelStdioRead(cpu: CpuState): Unit = UNIMPLEMENTED(0x3054D478)
    fun sceKernelRegisterStdoutPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0x432D8F5C)
    fun sceKernelRegisterStderrPipe(cpu: CpuState): Unit = UNIMPLEMENTED(0x6F797E03)
    fun sceKernelStdioOpen(cpu: CpuState): Unit = UNIMPLEMENTED(0x924ABA61)
    fun sceKernelStdioClose(cpu: CpuState): Unit = UNIMPLEMENTED(0x9D061C19)
    fun sceKernelStdioWrite(cpu: CpuState): Unit = UNIMPLEMENTED(0xA3B931DB)
    fun sceKernelStdioSendChar(cpu: CpuState): Unit = UNIMPLEMENTED(0xA46785C9)


    override fun registerModule() {
        registerFunctionInt("sceKernelStdin", 0x172D316E, since = 150) { sceKernelStdin() }
        registerFunctionInt("sceKernelStdout", 0xA6BAB2E9, since = 150) { sceKernelStdout() }
        registerFunctionInt("sceKernelStderr", 0xF78BA90A, since = 150) { sceKernelStderr() }

        registerFunctionRaw("sceKernelStdioLseek", 0x0CBB0571, since = 150) { sceKernelStdioLseek(it) }
        registerFunctionRaw("sceKernelStdioRead", 0x3054D478, since = 150) { sceKernelStdioRead(it) }
        registerFunctionRaw("sceKernelRegisterStdoutPipe", 0x432D8F5C, since = 150) { sceKernelRegisterStdoutPipe(it) }
        registerFunctionRaw("sceKernelRegisterStderrPipe", 0x6F797E03, since = 150) { sceKernelRegisterStderrPipe(it) }
        registerFunctionRaw("sceKernelStdioOpen", 0x924ABA61, since = 150) { sceKernelStdioOpen(it) }
        registerFunctionRaw("sceKernelStdioClose", 0x9D061C19, since = 150) { sceKernelStdioClose(it) }
        registerFunctionRaw("sceKernelStdioWrite", 0xA3B931DB, since = 150) { sceKernelStdioWrite(it) }
        registerFunctionRaw("sceKernelStdioSendChar", 0xA46785C9, since = 150) { sceKernelStdioSendChar(it) }
    }
}
