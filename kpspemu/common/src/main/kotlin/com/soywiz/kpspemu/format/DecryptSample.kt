package com.soywiz.kpspemu.format

import com.soywiz.korio.Korio
import com.soywiz.korio.vfs.LocalVfs

fun main(args: Array<String>) = Korio {
	val ebootBin = LocalVfs("c:/temp/1/EBOOT.BIN").readAll()
	//val decrypted1 = CryptedElf.decrypt(ebootBin)
	val decrypted2 = CryptedElf.decrypt(ebootBin)
	//LocalVfs("c:/temp/1/EBOOT.BIN.decrypted.kpspemu1").writeBytes(decrypted1)
	LocalVfs("c:/temp/1/EBOOT.BIN.decrypted.kpspemu2").writeBytes(decrypted2)
}