package com.soywiz.kpspemu.format

import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.stream.*

class Pbp(val version: Int, val base: AsyncStream, val streams: List<AsyncStream>) {
	val streamsByName = NAMES.zip(streams).toMap()

	companion object {
		const val PBP_MAGIC = 0x50425000

		const val PARAM_SFO = "param.sfo"
		const val ICON0_PNG = "icon0.png"
		const val ICON1_PMF = "icon1.pmf"
		const val PIC0_PNG = "pic0.png"
		const val PIC1_PNG = "pic1.png"
		const val SND0_AT3 = "snd0.at3"
		const val PSP_DATA = "psp.data"
		const val PSAR_DATA = "psar.data"

		val NAMES = listOf(PARAM_SFO, ICON0_PNG, ICON1_PMF, PIC0_PNG, PIC1_PNG, SND0_AT3, PSP_DATA, PSAR_DATA)

		suspend fun check(s: AsyncStream): Boolean {
			return s.duplicate().readS32_le() == PBP_MAGIC
		}

		suspend fun load(s: AsyncStream): Pbp {
			val magic = s.readS32_le()
			if (magic != PBP_MAGIC) invalidOp("Not a PBP file")
			val version = s.readS32_le()
			val offsets = s.readIntArray_le(8).toList() + listOf(s.size().toInt())
			val streams = (0 until (offsets.size - 1)).map { s.sliceWithBounds(offsets[it].toLong(), offsets[it + 1].toLong()) }
			return Pbp(version, s, streams)
		}
	}

	operator fun get(name: String) = streamsByName[name.toLowerCase()]?.duplicate()
	operator fun get(index: Int) = streams[index]
}
