package com.soywiz.kpspemu.kirk

import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.stream.*
import com.soywiz.korio.util.IdEnum
import com.soywiz.kpspemu.util.*

object Kirk {
	fun hleUtilsBufferCopyWithRange(output: SyncStream, input: SyncStream, command: CommandEnum): Unit = when (command) {
		CommandEnum.DECRYPT_PRIVATE -> kirk_CMD1(output, input)
		CommandEnum.ENCRYPT_IV_0 -> kirk_CMD4(output, input)
		CommandEnum.DECRYPT_IV_0 -> kirk_CMD7(output, input)
		CommandEnum.PRIV_SIG_CHECK -> kirk_CMD10(input)
		CommandEnum.SHA1_HASH -> kirk_CMD11(output, input)
		CommandEnum.ECDSA_GEN_KEYS -> kirk_CMD12(output)
		CommandEnum.ECDSA_MULTIPLY_POINT -> kirk_CMD13(output, input)
		CommandEnum.PRNG -> kirk_CMD14(output, TODO("argument outSize"))
		CommandEnum.ECDSA_SIGN -> kirk_CMD16(output, input)
		CommandEnum.ECDSA_VERIFY -> kirk_CMD17(input)
		else -> TODO("Not implemented hleUtilsBufferCopyWithRange! with command $command: $command")
	}

	fun kirk_CMD7(output: SyncStream, input: SyncStream) {
		output.clone().writeBytes(CMD7(input.slice()))
	}

	fun kirk_CMD1(output: SyncStream, input: SyncStream) {
		//console.log(input.sliceWithLength(0, 128).readAllBytes());
		val header = input.slice().read(AES128CMACHeader)
		if (header.Mode != KirkMode.Cmd1) throw invalidOp("Kirk mode != Cmd1")
		val Keys = AES.decryptAes128Cbc(input.slice().readBytes(32), KirkKeys.kirk1_key)
		val KeyAes = Keys.copyOfRange(0, 16)
		var KeyCmac = Keys.copyOfRange(16, 32)
		val PaddedDataSize = (header.DataSize + 15) and (-16)
		val Output = AES.decryptAes128Cbc(input.slice().skip(header.DataOffset + AES128CMACHeader.size).readBytes(PaddedDataSize), KeyAes)
		output.write(Output, 0, header.DataSize)
	}

	fun kirk_CMD17(input: SyncStream): Unit = TODO()
	fun kirk_CMD16(output: SyncStream, input: SyncStream): Unit = TODO()
	fun kirk_CMD14(output: SyncStream, outsize: Any): Unit = TODO()
	fun kirk_CMD13(output: SyncStream, input: SyncStream): Unit = TODO()
	fun kirk_CMD12(output: SyncStream): Unit = TODO()
	fun kirk_CMD11(output: SyncStream, input: SyncStream): Unit = TODO()
	fun kirk_CMD4(output: SyncStream, input: SyncStream): Unit = TODO()
	fun kirk_CMD10(input: SyncStream): Unit = TODO()

	fun CMD7(input: SyncStream): ByteArray {
		val header = input.read(KIRK_AES128CBC_HEADER)
		if (header.mode != KirkMode.DecryptCbc) throw Error("Kirk Invalid mode '" + header.mode + "'")
		if (header.data_size == 0) invalidOp("Kirk data size == 0")
		return AES.decryptAes128Cbc(input.readAll(), getKirk7Key(header.keyseed))
		//return AES.decryptAes128Cbc(input.readBytes(header.data_size), kirk_4_7_get_key(header.keyseed))
	}

	fun getKirk7Key(key_type: Int): ByteArray = KirkKeys.kirk7_keys[key_type] ?: invalidOp("Unsupported key $key_type")

	enum class KirkMode(override val id: Int) : IdEnum {
		Invalid0(0), Cmd1(1), Cmd2(2), Cmd3(3), EncryptCbc(4), DecryptCbc(5);
		companion object : UINT8_ENUM<KirkMode>(values())
	}

	enum class CommandEnum(override val id: Int) : IdEnum {
		DECRYPT_PRIVATE(0x1), // Master decryption command, used by firmware modules. Applies CMAC checking. Super-Duper decryption (no inverse) Private Sig + Cipher PSP_KIRK_CMD_DECRYPT_PRIVATE Code: 1, 0x01
		ENCRYPT_SIGN(0x2), // Used for key type 3 (blacklisting), encrypts and signs data with a ECDSA signature. Encrypt Operation (inverse of 0x03) Private Sig + Cipher Code: 2, 0x02
		DECRYPT_SIGN(0x3), // Used for key type 3 (blacklisting), decrypts and signs data with a ECDSA signature. Decrypt Operation (inverse of 0x02) Private Sig + Cipher Code: 3, 0x03
		ENCRYPT_IV_0(0x4), // Key table based encryption used for general purposes by several modules. Encrypt Operation (inverse of 0x07) (IV=0) Cipher KIRK_CMD_ENCRYPT_IV_0 Code: 4, 0x04
		ENCRYPT_IV_FUSE(0x5), // Fuse ID based encryption used for general purposes by several modules. Encrypt Operation (inverse of 0x08) (IV=FuseID) Cipher KIRK_CMD_ENCRYPT_IV_FUSE Code: 5, 0x05
		ENCRYPT_IV_USER(0x6), // User specified ID based encryption used for general purposes by several modules. Encrypt Operation (inverse of 0x09) (IV=UserDefined) Cipher KIRK_CMD_ENCRYPT_IV_USER Code: 6, 0x06
		DECRYPT_IV_0(0x7), // Key table based decryption used for general purposes by several modules.Decrypt Operation(inverse of 0x04) Cipher KIRK_CMD_DECRYPT_IV_0 Code: 7, 0x07
		DECRYPT_IV_FUSE(0x8), // Fuse ID based decryption used for general purposes by several modules. Decrypt Operation (inverse of 0x05) Cipher KIRK_CMD_DECRYPT_IV_FUSE Code: 8, 0x08
		DECRYPT_IV_USER(0x9), // User specified ID based decryption used for general purposes by several modules. Decrypt Operation (inverse of 0x06) Cipher PSP_KIRK_CMD_DECRYPT_IV_USER Code: 9, 0x09
		PRIV_SIG_CHECK(0xA), // Private signature (SCE) checking command. Private Signature Check (checks for private SCE sig) Sig Gens KIRK_CMD_PRIV_SIG_CHECK Code: 10, 0x0A
		SHA1_HASH(0xB), // SHA1 hash generating command. SHA1 Hash Sig Gens PSP_KIRK_CMD_SHA1_HASH Code: 11, 0x0B
		ECDSA_GEN_KEYS(0xC), // ECDSA key generating mul1 command.  Mul1 Sig Gens Code: 12, 0x0C
		ECDSA_MULTIPLY_POINT(0xD), // ECDSA key generating mul2 command.  Mul2 Sig Gens Code: 13, 0x0D
		PRNG(0xE), // Random number generating command.  Random Number Gen Sig Gens Code: 14, 0x0E
		INIT(0xF), // KIRK initialization command. (absolutely no idea? could be KIRK initialization) Sig Gens Code: 15, 0x0F
		ECDSA_SIGN(0x10), // ECDSA signing command. Signature Gen Code: 16, 0x10
		ECDSA_VERIFY(0x11), // ECDSA checking command. Signature Check (checks for generated sigs) Sig Checks Code: 17, 0x11
		CERT_VERIFY(0x12); // Certificate checking command. Certificate Check (idstorage signatures) Sig Checks Code: 18, 0x12

		companion object : UINT8_ENUM<CommandEnum>(values())
	}

	data class KIRK_AES128CBC_HEADER(
		var mode: KirkMode = KirkMode.Invalid0,
		var unk_4: Int = 0,
		var unk_8: Int = 0,
		var keyseed: Int = 0,
		var data_size: Int = 0
	) {
		companion object : Struct<KIRK_AES128CBC_HEADER>({ KIRK_AES128CBC_HEADER() },
			KIRK_AES128CBC_HEADER::mode AS INT32.asEnum(KirkMode),
			KIRK_AES128CBC_HEADER::unk_4 AS INT32,
			KIRK_AES128CBC_HEADER::unk_8 AS INT32,
			KIRK_AES128CBC_HEADER::keyseed AS INT32,
			KIRK_AES128CBC_HEADER::data_size AS INT32
		)
	}

	@Suppress("ArrayInDataClass")
	data class AES128CMACHeader(
		var AES_key: ByteArray = ByteArray(16),
		var CMAC_key: ByteArray = ByteArray(16),
		var CMAC_header_hash: ByteArray = ByteArray(16),
		var CMAC_data_hash: ByteArray = ByteArray(16),
		var Unknown1: ByteArray = ByteArray(32),
		var Mode: KirkMode = KirkMode.Invalid0,
		var UseECDSAhash: Int = 0,
		var Unknown2: ByteArray = ByteArray(14),
		var DataSize: Int = 0,
		var DataOffset: Int = 0,
		var Unknown3: ByteArray = ByteArray(8),
		var Unknown4: ByteArray = ByteArray(16)
	) { // SIZE: 0090
		companion object : Struct<AES128CMACHeader>({ AES128CMACHeader() },
			AES128CMACHeader::AES_key AS BYTEARRAY(16),
			AES128CMACHeader::CMAC_key AS BYTEARRAY(16),
			AES128CMACHeader::CMAC_header_hash AS BYTEARRAY(16),
			AES128CMACHeader::CMAC_data_hash AS BYTEARRAY(16),
			AES128CMACHeader::Unknown1 AS BYTEARRAY(32),
			AES128CMACHeader::Mode AS UINT8.asEnum(KirkMode),
			AES128CMACHeader::UseECDSAhash AS UINT8,
			AES128CMACHeader::Unknown2 AS BYTEARRAY(14),
			AES128CMACHeader::DataSize AS INT32,
			AES128CMACHeader::DataOffset AS INT32,
			AES128CMACHeader::Unknown3 AS BYTEARRAY(8),
			AES128CMACHeader::Unknown4 AS BYTEARRAY(16)
		)
	}
}