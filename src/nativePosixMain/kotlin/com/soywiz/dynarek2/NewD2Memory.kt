package com.soywiz.dynarek2

import kotlinx.cinterop.*
import platform.posix.*

actual val isNativeWindows: Boolean = false

actual fun NewD2Memory(size: Int): D2MemoryFreeable {
    return object : D2MemoryFreeable {
        val ptr = mmap(
            null, size.convert(),
            PROT_READ or PROT_WRITE or PROT_EXEC,
            MAP_ANONYMOUS or MAP_SHARED, -1, 0
        )?.reinterpret<ByteVar>() ?: error("Couldn't reserve memory")

        override val mem: D2Memory = D2Memory(ptr)
        //override val mem: D2Memory = ptr.reinterpret<ByteVar>()

        override fun free() {
            munmap(ptr, size.convert())
        }
    }
}
