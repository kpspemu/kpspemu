package com.soywiz.kpspemu.util

import com.soywiz.korio.file.*

object VfsUtil {
    fun normalize(path: String): String {
        return PathInfo(path).normalize()
    }
}