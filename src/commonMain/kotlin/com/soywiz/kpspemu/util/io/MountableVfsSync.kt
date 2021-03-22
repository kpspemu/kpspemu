package com.soywiz.kpspemu.util.io

import com.soywiz.klogger.*
import com.soywiz.korio.*
import com.soywiz.korio.file.*

fun MountableVfsSyncNew(callback: MountableSync.() -> Unit): VfsFile {
    val logger = Logger("MountableVfsSync")

    val mount = object : Vfs.Proxy(), MountableSync {
        private val _mounts = ArrayList<Pair<String, VfsFile>>()

        override val mounts: Map<String, VfsFile> get() = _mounts.toMap()

        override fun mount(folder: String, file: VfsFile) = this.apply {
            _unmount(folder)
            _mounts += VfsUtil.normalize(folder) to file
            resort()
        }

        override fun unmount(folder: String): MountableSync = this.apply {
            _unmount(folder)
            resort()
        }

        override fun unmountAll(): MountableSync = this.apply {
            _mounts.clear()
        }

        private fun _unmount(folder: String) {
            _mounts.removeAll { it.first == VfsUtil.normalize(folder) }
        }

        private fun resort() {
            _mounts.sortBy { -it.first.length }
        }

        override suspend fun VfsFile.transform(): VfsFile {
            //return super.transform(out)
            return this
        }

        override suspend fun access(path: String): VfsFile {
            val rpath = VfsUtil.normalize(path)
            for ((base, file) in _mounts) {
                //println("$base/$file")
                if (rpath.startsWith(base)) {
                    val nnormalizedPath = rpath.substring(base.length)
                    val subpath = VfsUtil.normalize(nnormalizedPath).trim('/')
                    logger.warn { "Accessing $file : $subpath ($nnormalizedPath)" }
                    val res = file[subpath]
                    logger.warn { " --> $res (${res.exists()})" }
                    return res
                }
            }
            logger.warn { "Can't find $rpath in mounted ${_mounts.map { it.first }}" }
            throw FileNotFoundException(path)
        }
    }
    callback(mount)
    return mount.root
}


//inline fun MountableVfs(callback: Mountable.() -> Unit): VfsFile {
//	val mount = MountableVfs()
//	callback(mount)
//	return mount.root
//}

interface MountableSync {
    fun mount(folder: String, file: VfsFile): MountableSync
    fun unmount(folder: String): MountableSync
    fun unmountAll(): MountableSync
    val mounts: Map<String, VfsFile>
}

