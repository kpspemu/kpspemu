package com.soywiz.kpspemu.util

fun ByteArray.copyReinterpretedAsIntArray(): IntArray {
	val out = IntArray(size / 4)
	var m = 0
	for (n in 0 until out.size) {
		val v3 = this[m++].toInt() and 0xFF
		val v2 = this[m++].toInt() and 0xFF
		val v1 = this[m++].toInt() and 0xFF
		val v0 = this[m++].toInt() and 0xFF
		out[n] = (v0 shl 0) or (v1 shl 8) or (v2 shl 16) or (v3 shl 24)
	}
	return out
}

fun IntArray.copyReinterpretedAsByteArray(): ByteArray {
	val out = ByteArray(size * 4)
	var m = 0
	for (n in 0 until size) {
		val v = this[n]
		out[m++] = ((v shr 24) and 0xFF).toByte()
		out[m++] = ((v shr 16) and 0xFF).toByte()
		out[m++] = ((v shr 8) and 0xFF).toByte()
		out[m++] = ((v shr 0) and 0xFF).toByte()
	}
	return out
}

fun IntArray.copyReinterpretedAsByteArrayRev(): ByteArray {
	val out = ByteArray(size * 4)
	var m = 0
	for (n in 0 until size) {
		val v = this[n]
		out[m++] = ((v shr 0) and 0xFF).toByte()
		out[m++] = ((v shr 8) and 0xFF).toByte()
		out[m++] = ((v shr 16) and 0xFF).toByte()
		out[m++] = ((v shr 24) and 0xFF).toByte()
	}
	return out
}