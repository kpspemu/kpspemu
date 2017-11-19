package com.soywiz.kpspemu.format

import com.soywiz.kmem.UByteArray
import com.soywiz.kmem.fill
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.stream.*
import com.soywiz.korio.util.toUnsigned
import com.soywiz.kpspemu.kirk.Kirk
import com.soywiz.kpspemu.util.*

object CryptedElf {
	suspend fun decrypt(input: ByteArray): ByteArray {
		val out = decrypt1(input.openSync()).readAll()
		//LocalVfs("c:/temp/decrypted.bin").writeBytes(out)
		return out
	}

	fun decrypt(input: SyncStream) = decrypt1(input.clone())

	fun getTagInfo(checkTag: Int): CryptedPrxKeys144.TagInfo?
		= CryptedPrxKeys144.g_tagInfo.firstOrNull { it.tag == checkTag }

	fun getTagInfo2(checkTag: Int): CryptedPrxKeys16.TagInfo2?
		= CryptedPrxKeys16.g_tagInfo2.firstOrNull { it.tag == checkTag }

	fun copyFromTo(from: ByteArray, fromOffset: Int, to: ByteArray, toOffset: Int, count: Int) {
		for (n in 0 until count) to[toOffset + n] = from[fromOffset + n]
	}

	fun memset(array: ByteArray, offset: Int, count: Int, value: Int) {
		for (n in 0 until count) array[offset + n] = value.toByte()
	}

	fun decrypt1(pbIn: SyncStream): SyncStream {
		val cbTotal = pbIn.length
		val _pbOut = ByteArray(cbTotal.toInt())
		val _pbOutU = UByteArray(_pbOut)
		val pbOut = _pbOut.openSync()
		pbOut.slice().writeStream(pbIn)

		val header = Header.read(pbIn.slice())
		val pti = getTagInfo(header.tag) ?: invalidOp("Can't find tag ${header.tag}")

		// build conversion into pbOut
		pbOut.slice().writeStream(pbIn)
		pbOut.slice().writeBytes(ByteArray(0x150).apply { fill(0) })
		pbOut.slice().writeBytes(ByteArray(0x40).apply { fill(0x55) })

		// step3 demangle in place
		//kirk.KIRK_AES128CBC_HEADER.struct.write();
		val h7_header = Kirk.KIRK_AES128CBC_HEADER()
		h7_header.mode = Kirk.KirkMode.DecryptCbc
		h7_header.unk_4 = 0
		h7_header.unk_8 = 0
		h7_header.keyseed = pti.code // initial seed for PRX
		h7_header.data_size = 0x70 // size

		Kirk.KIRK_AES128CBC_HEADER.write(pbOut.sliceWithStart(0x2C), h7_header)

		// redo part of the SIG check (step2)
		val buffer1 = MemorySyncStream(ByteArray(0x150))
		buffer1.sliceWithStart(0x00).writeStream(pbIn.sliceWithSize(0xD0, 0x80))
		buffer1.sliceWithStart(0x80).writeStream(pbIn.sliceWithSize(0x80, 0x50))
		buffer1.sliceWithStart(0xD0).writeStream(pbIn.sliceWithSize(0x00, 0x80))

		//console.log('buffer1', buffer1.slice().readAllBytes());

		if (pti.codeExtra != 0) {
			val buffer2 = MemorySyncStream(ByteArray(20 + 0xA0))

			// KIRK_AES128CBC_HEADER
			val bb = buffer2.slice()
			bb.write32_le(5)
			bb.write32_le(0)
			bb.write32_le(0)
			bb.write32_le(pti.codeExtra)
			bb.write32_le(0xA0)

			bb.writeStream(buffer1.sliceWithSize(0x10, 0xA0))

			Kirk.hleUtilsBufferCopyWithRange(
				buffer2.sliceWithSize(0, 20 + 0xA0),
				buffer2.sliceWithSize(0, 20 + 0xA0),
				Kirk.CommandEnum.DECRYPT_IV_0
			)

			// copy result back
			buffer1.slice().writeStream(buffer2.sliceWithSize(0, 0xA0))
		}

		pbOut.sliceWithStart(0x40).writeStream(buffer1.sliceWithSize(0x40, 0x40))

		for (iXOR in 0 until 0x70) {
			_pbOutU[0x40 + iXOR] = ((_pbOutU[0x40 + iXOR] xor pti.skey[0x14 + iXOR].toUnsigned()) and 0xFF)
		}

		Kirk.hleUtilsBufferCopyWithRange(
			pbOut.sliceWithSize(0x2C, 20 + 0x70),
			pbOut.sliceWithSize(0x2C, 20 + 0x70),
			Kirk.CommandEnum.DECRYPT_IV_0
		)

		var iXOR = 0x6F
		while (iXOR >= 0) {
			_pbOutU[0x40 + iXOR] = (_pbOutU[0x2C + iXOR] xor pti.skey[0x20 + iXOR].toUnsigned()) and 0xFF
			iXOR--
		}

		pbOut.sliceWithStart(0x80).writeBytes(ByteArray(0x30).apply { fill(0) })

		_pbOutU[0xA0] = 1
		// copy unscrambled parts from header
		pbOut.sliceWithStart(0xB0).writeStream(pbIn.sliceWithSize(0xB0, 0x20)) // file size + lots of zeros
		pbOut.sliceWithStart(0xD0).writeStream(pbIn.sliceWithSize(0x00, 0x80)) // ~PSP header

		//for (n in 0 until 0x100) println("%d: %d".format(n, _pbOut[n]))

		// step4: do the actual decryption of code block
		//  point 0x40 bytes into the buffer to key info
		Kirk.hleUtilsBufferCopyWithRange(
			pbOut.sliceWithSize(0x00, cbTotal),
			pbOut.sliceWithSize(0x40, cbTotal - 0x40),
			Kirk.CommandEnum.DECRYPT_PRIVATE
		)

		//File.WriteAllBytes("../../../TestInput/temp.bin", _pbOut);

		val outputSize = pbIn.sliceWithStart(0xB0).readS32_le()

		return pbOut.sliceWithSize(0, outputSize)
	}

	@Suppress("ArrayInDataClass")
	data class Header(
		var magic: Int = 0,
		var modAttr: Int = 0,
		var compModAttr: Int = 0,
		var modVerLo: Int = 0,
		var modVerHi: Int = 0,
		var moduleName: String = "",
		var modVersion: Int = 0,
		var nsegments: Int = 0,
		var elfSize: Int = 0,
		var pspSize: Int = 0,
		var bootEntry: Int = 0,
		var modInfoOffset: Int = 0,
		var bssSize: Int = 0,
		var segAlign: CharArray = CharArray(4),
		var segAddress: IntArray = IntArray(4),
		var segSize: IntArray = IntArray(4),
		var reserved: IntArray = IntArray(5),
		var devkitVersion: Int = 0,
		var decMode: Int = 0,
		var pad: Int = 0,
		var overlapSize: Int = 0,
		var aesKey: ByteArray = ByteArray(16),
		var cmacKey: ByteArray = ByteArray(16),
		var cmacHeaderHash: ByteArray = ByteArray(16),
		var compressedSize: Int = 0,
		var compressedOffset: Int = 0,
		var unk1: Int = 0,
		var unk2: Int = 0,
		var cmacDataHash: ByteArray = ByteArray(16),
		var tag: Int = 0,
		var sigcheck: ByteArray = ByteArray(88),
		var sha1Hash: ByteArray = ByteArray(20),
		var keyData: ByteArray = ByteArray(16)
	) {
		companion object : Struct<Header>({ Header() },
			Header::magic AS INT32,              // 0000
			Header::modAttr AS UINT16,           // 0004
			Header::compModAttr AS UINT16,       // 0006
			Header::modVerLo AS UINT8,           // 0008
			Header::modVerHi AS UINT8,           // 0009
			Header::moduleName AS STRINGZ(28),   // 000A
			Header::modVersion AS UINT8,         // 0026
			Header::nsegments AS UINT8,          // 0027
			Header::elfSize AS INT32,            // 0028
			Header::pspSize AS INT32,            // 002C
			Header::bootEntry AS INT32,          // 0030
			Header::modInfoOffset AS INT32,      // 0034
			Header::bssSize AS INT32,            // 0038
			Header::segAlign AS CHARARRAY(4),    // 003C
			Header::segAddress AS INTARRAY(4),   // 0044
			Header::segSize AS INTARRAY(4),      // 0054
			Header::reserved AS INTARRAY(5),     // 0064
			Header::devkitVersion AS INT32,
			Header::decMode AS UINT8,
			Header::pad AS UINT8,
			Header::overlapSize AS UINT16,
			Header::aesKey AS BYTEARRAY(16),
			Header::cmacKey AS BYTEARRAY(16),
			Header::cmacHeaderHash AS BYTEARRAY(16),
			Header::compressedSize AS INT32,
			Header::compressedOffset AS INT32,
			Header::unk1 AS INT32,
			Header::unk2 AS INT32,
			Header::cmacDataHash AS BYTEARRAY(16),
			Header::tag AS INT32,
			Header::sigcheck AS BYTEARRAY(88),
			Header::sha1Hash AS BYTEARRAY(20),
			Header::keyData AS BYTEARRAY(16)
		)
	}
}