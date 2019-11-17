package com.soywiz.korio.file

import com.soywiz.korio.lang.*

val VfsFile.basename get() = this.baseName
val VfsFile.fullname get() = this.fullName

object VfsUtil {
    fun normalize(path: String): String {
        return PathInfo(path).normalize()
    }
}

typealias FileNotFoundException = com.soywiz.korio.lang.FileNotFoundException
