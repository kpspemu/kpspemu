package com.soywiz.kpspemu.util.io

import com.soywiz.klogger.Logger
import com.soywiz.korio.FileNotFoundException
import com.soywiz.korio.vfs.Vfs
import com.soywiz.korio.vfs.VfsFile
import com.soywiz.korio.vfs.VfsUtil

fun MountableVfsSync(callback: MountableSync.() -> Unit): VfsFile {
	val logger = Logger("MountableVfsSync")

	val mount = object : Vfs.Proxy(), MountableSync {
		private val _mounts = ArrayList<Pair<String, VfsFile>>()

		override val mounts: Map<String, VfsFile> get() = _mounts.toMap()

		override fun mount(folder: String, file: VfsFile) = this.apply {
			_unmount(folder)
			_mounts += com.soywiz.korio.vfs.VfsUtil.normalize(folder) to file
			resort()
		}

		override fun unmount(folder: String): MountableSync = this.apply {
			_unmount(folder)
			resort()
		}

		private fun _unmount(folder: String) {
			_mounts.removeAll { it.first == VfsUtil.normalize(folder) }
		}

		private fun resort() {
			_mounts.sortBy { -it.first.length }
		}

		suspend override fun transform(out: VfsFile): VfsFile {
			//return super.transform(out)
			return out
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
	val mounts: Map<String, VfsFile>
}

