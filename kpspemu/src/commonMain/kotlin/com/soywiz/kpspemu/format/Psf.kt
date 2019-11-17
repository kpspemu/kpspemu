package com.soywiz.kpspemu.format

import com.soywiz.korio.error.*
import com.soywiz.korio.lang.*
import com.soywiz.korio.stream.*
import com.soywiz.korio.lang.invalidOp as invalidOp1

class Psf {
    data class DataType(val id: Int) {
        companion object {
            val BINARY = DataType(0)
            val TEXT = DataType(2)
            val INT = DataType(4)
        }
    }

    class HeaderStruct(
        val magic: Int,
        val version: Int,
        val keyTable: Int,
        val valueTable: Int,
        val numberOfPairs: Int
    ) {
        companion object {
            fun read(s: SyncStream): HeaderStruct = HeaderStruct(
                magic = s.readS32_le(),
                version = s.readS32_le(),
                keyTable = s.readS32_le(),
                valueTable = s.readS32_le(),
                numberOfPairs = s.readS32_le()
            )
        }
    }

    interface IEntryStruct {
        val key: String
        val value: Any?
    }

    class EntryStruct(
        val keyOffset: Int,
        private val unknown: Int,
        val dataType: DataType,
        val valueSize: Int,
        val valueSizePad: Int,
        val valueOffset: Int
    ) : IEntryStruct {
        override var key: String = ""
        override var value: Any? = null

        companion object {
            fun read(s: SyncStream): EntryStruct = EntryStruct(
                keyOffset = s.readU16_le(),
                unknown = s.readU8(),
                dataType = DataType(s.readU8()),
                valueSize = s.readS32_le(),
                valueSizePad = s.readS32_le(),
                valueOffset = s.readS32_le()
            )
        }

        override fun toString(): String = "Entry($key, $value)"
    }

    companion object {
        suspend operator fun invoke(stream: AsyncStream): Psf = Psf().apply { load(stream.readAll().openSync()) }
        operator fun invoke(stream: SyncStream): Psf = Psf().apply { load(stream) }
        operator fun invoke(bytes: ByteArray): Psf = Psf().apply { load(bytes.openSync()) }
        fun fromStream(stream: SyncStream): Psf = Psf().apply { load(stream) }
    }

    lateinit var entries: List<EntryStruct>; private set
    lateinit var entriesByName: Map<String, Any?>; private set
    lateinit var header: HeaderStruct; private set

    fun getString(key: String): String? = entriesByName[key]?.toString()
    fun getInt(key: String): Int? = entriesByName[key]?.toString()?.toInt()
    fun getDouble(key: String): Double? = entriesByName[key]?.toString()?.toDouble()

    fun load(stream: SyncStream) {
        val header = HeaderStruct.read(stream)
        this.header = header
        if (header.magic != 0x46535000) invalidOp1("Not a PSF file")
        val entries = (0 until header.numberOfPairs).map { EntryStruct.read(stream) }
        val entriesByName = LinkedHashMap<String, Any?>()

        val keysStream = stream.sliceStart(header.keyTable.toLong())
        val valuesStream = stream.sliceStart(header.valueTable.toLong())

        for (entry in entries) {
            val key = keysStream.sliceStart(entry.keyOffset.toLong()).readStringz(UTF8)
            val valueStream = valuesStream.sliceWithSize(entry.valueOffset.toLong(), entry.valueSize.toLong())
            entry.key = key

            when (entry.dataType) {
                DataType.BINARY -> entry.value = valueStream.readSlice(0)
                DataType.INT -> entry.value = valueStream.readS32_le()
                DataType.TEXT -> entry.value = valueStream.readStringz(UTF8)
                else -> invalidOp1("Unknown dataType: ${entry.dataType}")
            }

            entriesByName[entry.key] = entry.value
        }

        this.entries = entries
        this.entriesByName = entriesByName
    }
}
