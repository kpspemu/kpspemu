package com.soywiz.kpspemu.util

// WIP
/*
fun <T> MutableList<T>.shrinkTo(size: Int) {
	while (this.size > size) this.removeAt(this.size - 1)
}

class SortedMutableList<T>

class SparseMemory {
	data class Chunk(val position: Long, val data: ByteArray) {
		val low = position
		val high = position + data.size
		val range = low until high

		fun canConcatenateTo(right: Chunk): Boolean {
			return this.high == right.low
		}

		fun concatenateTo(right: Chunk): Chunk {
			if (!canConcatenateTo(right)) invalidOp("Can't concatenate")
			return Chunk(position, data + right.data)
		}
	}

	val chunks = arrayListOf<Chunk>()

	private fun sortChunks() {
		chunks.sortBy { it.position }
	}

	fun getChunkAt(position: Long): Chunk? {
		// @TODO: Performance. Use: chunks.binarySearch {}
		return chunks.firstOrNull { it.position == position }
	}

	fun getChunkContaining(position: Long): Chunk? {
		// @TODO: Performance. Use: chunks.binarySearch {}
		return chunks.firstOrNull { position in it.range }
	}

	fun getOverlappingChunks(position: Long, size: Int): List<Chunk> {
		val range = position until (position + size)
		return chunks.filter { it.range.overlapsWith(range) }
	}

	fun getRanges(position: Long, size: Int): List<LongRange> {
		val fullRange = position until (position + size)
		val chunks = getOverlappingChunks(position, size)
		val out = arrayListOf<LongRange>()
		for (n in 0 until chunks.size) {
			val l = chunks[n]
			val r = chunks.getOrNull(n + 1)
			out += l.range
			if (r == null) {
				out += l.high .. fullRange.endInclusive
			} else {
				out += l.high until r.low
			}
		}
		return out.filter { !it.isEmpty() }
	}

	fun mergeChunks() {
		sortChunks()
		var l = 0
		var r = 1
		while (r < chunks.size) {
			val lchunk = chunks[l]
			val rchunk = chunks[r]
			if (lchunk.canConcatenateTo(rchunk)) {
				chunks[l] = lchunk.concatenateTo(rchunk)
				r++
			} else {
				l++
				r++
			}
		}
		if (chunks.size >= 2) {
			chunks.shrinkTo(l + 1)
		}
	}

	fun write(position: Long, data: ByteArray, offset: Int = 0, len: Int = data.size - offset) {
		val ranges = getRanges(position, len)
		for (range in ranges) {
			val chunk = getChunkContaining(range.start)
			val roffset = range.start - position

			if (chunk != null) {
				arraycopy(data, offset + roffset, chunk.data, ) chunk.data
			} else {

			}
		}
		mergeChunks()
	}

	fun read(position: Long, data: ByteArray, offset: Int = 0, len: Int = data.size - offset) {

	}
}

class PatchedAsyncBaseStream(val base: AsyncStream, val patch: SparseMemory) : AsyncStreamBase() {

}

fun AsyncStream.patchWith(sparse: SparseMemory) = PatchedAsyncBaseStream(this, sparse).toAsyncStream()
*/
