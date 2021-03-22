package com.soywiz.dynarek2

import kotlinx.cinterop.*
import platform.posix.*
import platform.windows.*

actual val isNativeWindows: Boolean = true

actual fun NewD2Memory(size: Int): D2MemoryFreeable {
    return object : D2MemoryFreeable {
        val ptr = VirtualAlloc(
            null,
            size.convert(),
            (MEM_COMMIT or MEM_RESERVE).convert(), // 0x00001000 | 0x00002000
            PAGE_EXECUTE_READWRITE.convert() // 0x40
        )?.reinterpret<ByteVar>() ?: error("Couldn't reserve memory: size=$size, lastError=${GetLastError()}")

        init {
            memset(ptr, 0, size.convert())
        }

        override val mem: D2Memory = D2Memory(ptr)
        //override val mem: D2Memory = ptr.reinterpret<ByteVar>()

        override fun free() {
            VirtualFree(
                ptr.reinterpret(),
                size.convert(),
                (MEM_DECOMMIT or MEM_RELEASE).convert() // 0x4000 | 0x8000
            )
        }
    }
}



//private fun Int.nextAlignedTo(align: Int) = when {
//    align == 0 -> this
//    (this % align) == 0 -> this
//    else -> (((this / align) + 1) * align)
//}


/*
VirtualAlloc(
_In_opt_ LPVOID lpAddress,
_In_     SIZE_T dwSize,
_In_     DWORD  flAllocationType,
_In_     DWORD  flProtect
);
VirtualFree(
  _In_ LPVOID lpAddress,
  _In_ SIZE_T dwSize,
  _In_ DWORD  dwFreeType
);
 */
