package com.soywiz.korio.file

val VfsFile.basename get() = this.baseName
val VfsFile.fullname get() = this.fullName

object VfsUtil {
    fun normalize(path: String): String {
        return PathInfo(path).normalize()
    }

    fun combine(base: String, component: String): String {
        return PathInfo(base).combine(PathInfo(component)).fullPath
    }
}

typealias FileNotFoundException = com.soywiz.korio.lang.FileNotFoundException
