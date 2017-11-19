package com.soywiz.kpspemu.format

import com.soywiz.kmem.UByteArray
import com.soywiz.kmem.fill
import com.soywiz.korio.crypto.Hex
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
			Header::magic AS INT32,
			Header::modAttr AS UINT16,
			Header::compModAttr AS UINT16,
			Header::modVerLo AS UINT8,
			Header::modVerHi AS UINT8,
			Header::moduleName AS STRINGZ(28),
			Header::modVersion AS UINT8,
			Header::nsegments AS UINT8,
			Header::elfSize AS INT32,
			Header::pspSize AS INT32,
			Header::bootEntry AS INT32,
			Header::modInfoOffset AS INT32,
			Header::bssSize AS INT32,
			Header::segAlign AS CHARARRAY(4),
			Header::segAddress AS INTARRAY(4),
			Header::segSize AS INTARRAY(4),
			Header::reserved AS INTARRAY(5),
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


	@Suppress("RemoveRedundantCallsOfConversionMethods", "MemberVisibilityCanPrivate", "unused")
	object CryptedPrxKeys144 {
		@Suppress("ArrayInDataClass")
		data class TagInfo(val tag: Int, val skey: ByteArray, val code: Int, val codeExtra: Int)

		val g_tagInfo = listOf(
			TagInfo(tag = 0x00000000.toInt(), skey = Hex.decode("bef3217b1d5e9c29715e9c1c4546cb96e01b9b3c3dde85eb22207f4aaa6e20c265320bd5670577554008083cf2551d98f3f6d85fc5b08eee52814d94518627f8faba052733e52084e94a152732aa194840aaa35965cfb32c6d4674f20556653a8ff8b021268db1c55190c1644ec969d6f23570e809593a9d02714e6fce46a9dc1b881684a597b0bac625912472084cb3"), code = 0x42, codeExtra = 0x00),
			TagInfo(tag = 0x02000000.toInt(), skey = Hex.decode("32a9fdcc766fc051cfcc6d041e82e1494c023b7d6558da9d25988cccb57de9d1cbd8746887c97134fcb3ed725d36c8813ae361e159db92fcecb10920e44ca9b16b69032fd836e287e98c2b3b84e70503830871f939db39b037ea3b8905684de7bd385c2a13c88db07523b3152545be4690fd0301a2870ea96aa6ab52807bbf8563cee845d316d74d2d0de3f556e43aaf"), code = 0x45, codeExtra = 0x00),
			TagInfo(tag = 0x03000000.toInt(), skey = Hex.decode("caf5c8a680c0676d3a4d4f926aa07c049702640858a7d44f875a68bdc201279b352ab6833c536b720cfa22e5b4064bc2ac1c9d457b41c5a8a262ea4f42d71506098d623014ab4fc45e71ff697d83d8d28b0bedbeae576e1e02c4e861067a36be5e2b3f5458c03edb752085becc4d7e1e55ea6415b42578ecad8c53c07f2cf770d0c3e849c57ea9eda4b092f42ab05ee0"), code = 0x46, codeExtra = 0x00),
			TagInfo(tag = 0x4467415D.toInt(), skey = Hex.decode("05e080ef9f68543acd9cc943be27771b3800b85c62fe2edd2cf969f3c5940f1619005629c5103cbf6655cef226c6a2ce6f8101b61e48e764bdde340cb09cf298d704c53ff039fbc8d8b32102a236f96300483a9ae332cc6efd0c128e231636b089e6e1aeeb0255741cc6a6e4b43ef2741358fad7eb1619b057843212d297bcd2d8256464a5808332b18ada43c92a124b"), code = 0x59, codeExtra = 0x59),
			TagInfo(tag = 0x207BBF2F.toInt(), skey = Hex.decode("0008b533cd5f2ff31f88143c952a8a6ed5effe29e3ea941343d46bbd83c02108d379b3fa65e113e6d354a7f552298b10151e4b0abadeea61df6575550153463bc3ec54ae0933426119ffc970ece50a5b26f19d985f7a989d0e75bc5527ba6ec6e888e92dda0066f7cbdc8203f2f569556212438ed3e38f2887216f659c2ed137b49e532f8e9992a4f75839ed2365e939"), code = 0x5A, codeExtra = 0x5A),
			TagInfo(tag = 0x3ACE4DCE.toInt(), skey = Hex.decode("697087671756bd3adcb13ac27d5057ab407f6a06b9f9de24e459f706b124f5dc5e3e79132d025903a2e1e7aafab2b9764003169aba2f8287bb8fe219028a339e9a7e00d8f17a31eade7106637cca670baf92518626353cea8e8c442b5492598bcbe90246da6ce14dbbd564e18ed8ec07f8e5ff99c1008876ed91b05334740484bcdb26b4bb48f9365821144692b49b74"), code = 0x5B, codeExtra = 0x5B),
			TagInfo(tag = 0x07000000.toInt(), skey = Hex.decode("af00cb762fe61c11367eb2b7f9e88d6d6af14bd57303e9d982d999750e2bf851ad03616335bc408e942c332fe9aa13f5e9fe2ad2873934040cb85bfc899d3412bb81a414e83aed254f0e507d57b7d143adfd597b34bffb4c3674d1c3db21dac1808c4da35d232b964805423efe9fcf095c3f88d4b59c0ed9e9f4ae00e96d88f05b8aa5624655a552b5411997ac9fb7f5"), code = 0x4A, codeExtra = 0x00),
			TagInfo(tag = 0x08000000.toInt(), skey = Hex.decode("ef69cb1812898e15bb0ef9de23fbb04c18ee87366e4a8d8656c7b5191d5516ee6c2dcbe760c647973f1495ce77f45629de4a8203f19d0c2124eb29509fe6df81009bc839918b0cb0c2f92deffc933ae1a8a4948b9dd01d490d406a68e4c7d4cec9b7c89628dcaa1e840b17a4dc5d5d50cfc3a65d2dfa5d0eb519796ec7295ece94dbacaadd0cf7452537a7623d56e6cc"), code = 0x4B, codeExtra = 0x00),
			TagInfo(tag = 0xC0CB167C.toInt(), skey = Hex.decode("fa368eda4774d95d7498c176af7ee597bd09ab1cc6ba35988192d303cf05b20334e7822863f614e775276eb9c7af8abd29ecd31d6ca1a4ec87ec695f921e988521aefc7c16dde9ba0478a9e6fc02ee2e3d8adf61640531dd49e197963b3f45c256841df9c86bda39f5fee5b3a393c589bc8a5cfb12720b6ccbd30de1a8b2d0984718d65f5723dcf06a160177683b5c0f"), code = 0x5D, codeExtra = 0x5D),
			TagInfo(tag = 0x0B000000.toInt(), skey = Hex.decode("bf3c60a5412448d7cc6457f60b06901f453ea74e92d151e58a5db7e76e505a462210fb405033272c44da96808e19479977ee8d272e065d7445fa48c1af822583da86db5fcec415cb2fc62425b1c32e6c9ee39b36c41febf71ace511ef43605d7d8394dc313fb1874e14dc8e33cf018b14e8d01a20d77d8e690f320574163f9178fa6a46028dd2713644c9405124c2c0c"), code = 0x4E, codeExtra = 0x00),
			TagInfo(tag = 0x0C000000.toInt(), skey = Hex.decode("2f10bf1a71d096d5b252c56f1f53f2d4d9cd25f003af9aafcf57cfe0c49454255e67037084c87b90e44e2d000d7a680b4fa43a9e81da8ff58cac26ec9db4c93a37c071344d83f3b01144dc1031ea32a26bfae5e2034b5945871c3ae4d1d9da310370cd08df2f9cfa251d895a34195c9be566f322324a085fd51655699fbe452205d76d4fa1b8b8c400a613bc3bfcb777"), code = 0x4F, codeExtra = 0x00),
			TagInfo(tag = 0x0F000000.toInt(), skey = Hex.decode("bcfe81a3c9d5b9998d0a566c959f3030cc4626795e4eb682ad51391ac42e180ab43161c48a0cc577c6165f322e94d102c48aa30ac60a942a2647036733b12de50721efd2901ec885ba64d1c81dce8dc375a28b940346b80d373647e2dafc74cd663d8e5822e8286d8b541e896df53cf566dbbd0baa86b2c44bbceb2bf41f26fc05e7b8925269eedce542045e217feb8b"), code = 0x52, codeExtra = 0x00),
			TagInfo(tag = 0x862648D1.toInt(), skey = Hex.decode("98d6bf1124b3f9d7274952dd865b21166dc34a5017b2435847daa0e5e7a173bb35db15293afd5c3705a970bbcaef2b279107962ebb9907eac8e65ab873f7cac941e60e259e4ae7065d894452a555674653af849a744102e11e03baeeceb980ed725f31bc7f062158583031e806e7d0d23e93d8e6b47fd1d7c49650503b0ba5fd3dae35468a9c48eb2d762d4231328b5a"), code = 0x52, codeExtra = 0x52),
			TagInfo(tag = 0x207BBF2F.toInt(), skey = Hex.decode("0008b533cd5f2ff31f88143c952a8a6ed5effe29e3ea941343d46bbd83c02108d379b3fa65e113e6d354a7f552298b10151e4b0abadeea61df6575550153463bc3ec54ae0933426119ffc970ece50a5b26f19d985f7a989d0e75bc5527ba6ec6e888e92dda0066f7cbdc8203f2f569556212438ed3e38f2887216f659c2ed137b49e532f8e9992a4f75839ed2365e939"), code = 0x5A, codeExtra = 0x5A),
			TagInfo(tag = 0x09000000.toInt(), skey = Hex.decode("e8531b72c6313efca2a25bf872acf03caba7ee54cbbf59596b83b854131343bccff29e98b236cef0f84cba9831c971e9c85d37a0a02fe50826d40dac01d6e457c7616ec58ab91aeff4f8d9d108a7e95f079df03e8c1a0cfa5cea1ea9c582f4580203802cc3f6e67ebbbb6affd0d01021887a29d3d31200987bc859dc9257dc7fa65d3fdb87b723fcd38e692212e880b6"), code = 0x4C, codeExtra = 0x00),
			TagInfo(tag = 0xBB67C59F.toInt(), skey = Hex.decode("c757a7943398d39f718350f8290b8b32dab9bc2cc6b91829ba504c94d0e7dcf166390c64083d0bc9ba17adf44bf8a06c677c76f75aa5d3a46a5c084a7170b26bfb388bfab831db3ff296718b4aed9bdb845b6251b481144c08f584f67047b430748eaa93bc79c5908dc86e240212052e2e8474c797d985a1dd3a2b7a6d5b83fe4d188f50134f4cebd393190ed2df96ba"), code = 0x5E, codeExtra = 0x5E)
		)
	}

	@Suppress("RemoveRedundantCallsOfConversionMethods", "unused", "MemberVisibilityCanPrivate")
	object CryptedPrxKeys16 {
		@Suppress("ArrayInDataClass")
		data class TagInfo2(val tag: Int, val key: ByteArray, val code: Int)

		val g_tagInfo2 = listOf(
			TagInfo2(tag = 0x380228F0.toInt(), key = Hex.decode("f28f75a73191ce9e75bd2726b4b40c32"), code = 0x5A),
			TagInfo2(tag = 0x4C942AF0.toInt(), key = Hex.decode("418a354f693adf04fd3946a25c2df221"), code = 0x43),
			TagInfo2(tag = 0x4C9428F0.toInt(), key = Hex.decode("f1bc1707aeb7c830d8349d406a8edf4e"), code = 0x43),
			TagInfo2(tag = 0x4C9429F0.toInt(), key = Hex.decode("6d72a4ba7fbfd1f1a9f3bb071bc0b366"), code = 0x43),
			TagInfo2(tag = 0x4C941DF0.toInt(), key = Hex.decode("1d13e95004733dd2e1dab9c1e67b25a7"), code = 0x43),
			TagInfo2(tag = 0x4C941CF0.toInt(), key = Hex.decode("d6bdce1e12af9ae66930deda88b8fffb"), code = 0x43),
			TagInfo2(tag = 0x457B1EF0.toInt(), key = Hex.decode("a35d51e656c801cae377bfcdff24da4d"), code = 0x5B),
			TagInfo2(tag = 0x457B0BF0.toInt(), key = Hex.decode("7b9472274ccc543baedf4637ac014d87"), code = 0x5B),
			TagInfo2(tag = 0x457B0CF0.toInt(), key = Hex.decode("ac34bab1978dae6fbae8b1d6dfdff1a2"), code = 0x5B),
			TagInfo2(tag = 0x4C9419F0.toInt(), key = Hex.decode("bae2a31207ff041b64a51185f72f995b"), code = 0x43),
			TagInfo2(tag = 0x4C9418F0.toInt(), key = Hex.decode("eb1b530b624932581f830af4993d75d0"), code = 0x43),
			TagInfo2(tag = 0x4C941FF0.toInt(), key = Hex.decode("2c8eaf1dff79731aad96ab09ea35598b"), code = 0x43),
			TagInfo2(tag = 0x4C9417F0.toInt(), key = Hex.decode("bae2a31207ff041b64a51185f72f995b"), code = 0x43),
			TagInfo2(tag = 0x4C9416F0.toInt(), key = Hex.decode("eb1b530b624932581f830af4993d75d0"), code = 0x43),
			TagInfo2(tag = 0x4C9414F0.toInt(), key = Hex.decode("45ef5c5ded81998412948fabe8056d7d"), code = 0x43),
			TagInfo2(tag = 0x4C9415F0.toInt(), key = Hex.decode("701b082522a14d3b6921f9710aa841a9"), code = 0x43),
			TagInfo2(tag = 0xD82310F0.toInt(), key = Hex.decode("9d09fd20f38f10690db26f00ccc5512e"), code = 0x51),
			TagInfo2(tag = 0xD8231EF0.toInt(), key = Hex.decode("4f445c62b353c430fc3aa45becfe51ea"), code = 0x51),
			TagInfo2(tag = 0xD82328F0.toInt(), key = Hex.decode("5daa72f226604d1ce72dc8a32f79c554"), code = 0x51),
			TagInfo2(tag = 0x4C9412F0.toInt(), key = Hex.decode("26380aaca5d874d132b72abf799e6ddb"), code = 0x43),
			TagInfo2(tag = 0x4C9413F0.toInt(), key = Hex.decode("53e7abb9c64a4b779217b5740adaa9ea"), code = 0x43),
			TagInfo2(tag = 0x457B10F0.toInt(), key = Hex.decode("7110f0a41614d59312ff7496df1fda89"), code = 0x5B),
			TagInfo2(tag = 0x4C940DF0.toInt(), key = Hex.decode("3c2b51d42d8547da2dca18dffe5409ed"), code = 0x43),
			TagInfo2(tag = 0x4C9410F0.toInt(), key = Hex.decode("311f98d57b58954532ab3ae389324b34"), code = 0x43),
			TagInfo2(tag = 0x4C940BF0.toInt(), key = Hex.decode("3b9b1a56218014ed8e8b0842fa2cdc3a"), code = 0x43),
			TagInfo2(tag = 0x457B0AF0.toInt(), key = Hex.decode("e8be2f06b1052ab9181803e3eb647d26"), code = 0x5B),
			TagInfo2(tag = 0x38020AF0.toInt(), key = Hex.decode("ab8225d7436f6cc195c5f7f063733fe7"), code = 0x5A),
			TagInfo2(tag = 0x4C940AF0.toInt(), key = Hex.decode("a8b14777dc496a6f384c4d96bd49ec9b"), code = 0x43),
			TagInfo2(tag = 0x4C940CF0.toInt(), key = Hex.decode("ec3bd2c0fac1eeb99abcffa389f2601f"), code = 0x43),
			TagInfo2(tag = 0xCFEF09F0.toInt(), key = Hex.decode("a241e839665bfabb1b2d6e0e33e5d73f"), code = 0x62),
			TagInfo2(tag = 0x457B08F0.toInt(), key = Hex.decode("a4608fababdea5655d433ad15ec3ffea"), code = 0x5B),
			TagInfo2(tag = 0x380208F0.toInt(), key = Hex.decode("e75c857a59b4e31dd09ecec2d6d4bd2b"), code = 0x5A),
			TagInfo2(tag = 0xCFEF08F0.toInt(), key = Hex.decode("2e00f6f752cf955aa126b4849b58762f"), code = 0x62),
			TagInfo2(tag = 0xCFEF07F0.toInt(), key = Hex.decode("7ba1e25a91b9d31377654ab7c28a10af"), code = 0x62),
			TagInfo2(tag = 0xCFEF06F0.toInt(), key = Hex.decode("9f671a7a22f3590baa6da4c68bd00377"), code = 0x62),
			TagInfo2(tag = 0x457B06F0.toInt(), key = Hex.decode("15076326dbe2693456082a934e4b8ab2"), code = 0x5B),
			TagInfo2(tag = 0x380206F0.toInt(), key = Hex.decode("563b69f729882f4cdbd5de80c65cc873"), code = 0x5A),
			TagInfo2(tag = 0xCFEF05F0.toInt(), key = Hex.decode("cafbbfc750eab4408e445c6353ce80b1"), code = 0x62),
			TagInfo2(tag = 0x457B05F0.toInt(), key = Hex.decode("409bc69ba9fb847f7221d23696550974"), code = 0x5B),
			TagInfo2(tag = 0x380205F0.toInt(), key = Hex.decode("03a7cc4a5b91c207fffc26251e424bb5"), code = 0x5A),
			TagInfo2(tag = 0x16D59E03.toInt(), key = Hex.decode("c32489d38087b24e4cd749e49d1d34d1"), code = 0x62),
			TagInfo2(tag = 0x76202403.toInt(), key = Hex.decode("f3ac6e7c040a23e70d33d82473392b4a"), code = 0x5B),
			TagInfo2(tag = 0x0F037303.toInt(), key = Hex.decode("72b439ff349bae8230344a1da2d8b43c"), code = 0x5A),
			TagInfo2(tag = 0x457B28F0.toInt(), key = Hex.decode("b1b37f76c3fb88e6f860d3353ca34ef3"), code = 0x5B),
			TagInfo2(tag = 0xADF305F0.toInt(), key = Hex.decode("1299705e24076cd02d06fe7eb30c1126"), code = 0x60),
			TagInfo2(tag = 0xADF306F0.toInt(), key = Hex.decode("4705d5e3561e819b092f06db6b1292e0"), code = 0x60),
			TagInfo2(tag = 0xADF308F0.toInt(), key = Hex.decode("f662396e26224dca026416997b9ae7b8"), code = 0x60),
			TagInfo2(tag = 0x8004FD03.toInt(), key = Hex.decode("f4aef4e186ddd29c7cc542a695a08388"), code = 0x5D),
			TagInfo2(tag = 0xD91605F0.toInt(), key = Hex.decode("b88c458bb6e76eb85159a6537c5e8631"), code = 0x5D),
			TagInfo2(tag = 0xD91606F0.toInt(), key = Hex.decode("ed10e036c4fe83f375705ef6a44005f7"), code = 0x5D),
			TagInfo2(tag = 0xD91608F0.toInt(), key = Hex.decode("5c770cbbb4c24fa27e3b4eb4b4c870af"), code = 0x5D),
			TagInfo2(tag = 0xD91609F0.toInt(), key = Hex.decode("d036127580562043c430943e1c75d1bf"), code = 0x5D),
			TagInfo2(tag = 0x2E5E10F0.toInt(), key = Hex.decode("9d5c5baf8cd8697e519f7096e6d5c4e8"), code = 0x48),
			TagInfo2(tag = 0x2E5E12F0.toInt(), key = Hex.decode("8a7bc9d6525888ea518360ca1679e207"), code = 0x48),
			TagInfo2(tag = 0x2E5E12F0.toInt(), key = Hex.decode("ffa468c331cab74cf123ff01653d2636"), code = 0x48),
			TagInfo2(tag = 0xD9160AF0.toInt(), key = Hex.decode("10a9ac16ae19c07e3b607786016ff263"), code = 0x5D),
			TagInfo2(tag = 0xD9160BF0.toInt(), key = Hex.decode("8383f13753d0befc8da73252460ac2c2"), code = 0x5D),
			TagInfo2(tag = 0xD91611F0.toInt(), key = Hex.decode("61b0c0587157d9fa74670e5c7e6e95b9"), code = 0x5D),
			TagInfo2(tag = 0xD91612F0.toInt(), key = Hex.decode("9e20e1cdd788dec0319b10afc5b87323"), code = 0x5D),
			TagInfo2(tag = 0xD91613F0.toInt(), key = Hex.decode("ebff40d8b41ae166913b8f64b6fcb712"), code = 0x5D),
			TagInfo2(tag = 0x0A35EA03.toInt(), key = Hex.decode("f948380c9688a7744f65a054c276d9b8"), code = 0x5E),
			TagInfo2(tag = 0x7B0505F0.toInt(), key = Hex.decode("2d86773a56a44fdd3c167193aa8e1143"), code = 0x5E),
			TagInfo2(tag = 0x7B0506F0.toInt(), key = Hex.decode("781ad28724bda296183f893672909285"), code = 0x5E),
			TagInfo2(tag = 0x7B0508F0.toInt(), key = Hex.decode("c97d3e0a54816ec7137499746218e7dd"), code = 0x5E),
			TagInfo2(tag = 0x279D08F0.toInt(), key = Hex.decode("c7277285aba7f7f04cc186cce37f17ca"), code = 0x61),
			TagInfo2(tag = 0x279D06F0.toInt(), key = Hex.decode("76409e08db9b3ba1478a968ef3f76292"), code = 0x61),
			TagInfo2(tag = 0x279D05F0.toInt(), key = Hex.decode("23dc3bb5a982d6ea63a36e2b2be9e154"), code = 0x61),
			TagInfo2(tag = 0xD66DF703.toInt(), key = Hex.decode("224357682f41ce654ca37cc6c4acf360"), code = 0x61),
			TagInfo2(tag = 0x279D10F0.toInt(), key = Hex.decode("12570d8a166d8706037dc88b62a332a9"), code = 0x61),
			TagInfo2(tag = 0x3C2A08F0.toInt(), key = Hex.decode("1e2e3849dad41608272ef3bc37758093"), code = 0x67)
		)
	}
}