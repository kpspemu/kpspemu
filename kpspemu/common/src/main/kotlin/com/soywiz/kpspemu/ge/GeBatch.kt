package com.soywiz.kpspemu.ge

data class GeBatch(
	val state: GeState,
	val primType: Int,
	val vertexCount: Int,
	val vertexType: Int,
	val vertices: ByteArray,
	val indices: ShortArray
)