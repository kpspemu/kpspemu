package com.soywiz.kpspemu.util.io

import com.soywiz.korio.FileNotFoundException
import com.soywiz.korio.vfs.Vfs
import com.soywiz.korio.vfs.VfsFile
import com.soywiz.korio.vfs.VfsUtil

fun MountableVfsSync(callback: MountableSync.() -> Unit): VfsFile {
	val mount = object : Vfs.Proxy(), MountableSync {
		private val mounts = ArrayList<Pair<String, VfsFile>>()

		override fun mount(folder: String, file: VfsFile) = this.apply {
			_unmount(folder)
			mounts += com.soywiz.korio.vfs.VfsUtil.normalize(folder) to file
			resort()
		}

		override fun unmount(folder: String): MountableSync = this.apply {
			_unmount(folder)
			resort()
		}

		private fun _unmount(folder: String) {
			mounts.removeAll { it.first == VfsUtil.normalize(folder) }
		}

		private fun resort() {
			mounts.sortBy { -it.first.length }
		}

		suspend override fun transform(out: VfsFile): VfsFile {
			//return super.transform(out)
			return out
		}

		override suspend fun access(path: String): VfsFile {
			val rpath = VfsUtil.normalize(path)
			for ((base, file) in mounts) {
				//println("$base/$file")
				if (rpath.startsWith(base)) {
					return file[rpath.substring(base.length)]
				}
			}
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
}

