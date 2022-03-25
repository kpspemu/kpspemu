package com.soywiz.kpspemu.kirk

import com.soywiz.kmem.*
import com.soywiz.korio.error.*
import com.soywiz.korio.lang.*
import com.soywiz.korio.stream.*
import com.soywiz.korio.util.*
import com.soywiz.kpspemu.util.*
import com.soywiz.krypto.*
import com.soywiz.korio.error.invalidOp as invalidOp2
import com.soywiz.korio.lang.invalidOp as invalidOp1

object Kirk {
    fun hleUtilsBufferCopyWithRange(
        output: p_u8,
        outputSize: Int,
        input: p_u8,
        inputSize: Int,
        command: CommandEnum
    ): Int {
        try {
            val o = output.openSync(outputSize)
            val i = input.openSync(inputSize)
            hleUtilsBufferCopyWithRange(o, i, command)
            return 0
        } catch (e: Throwable) {
            println("ERROR at: hleUtilsBufferCopyWithRange")
            e.printStackTrace()
            return -1
        }
    }

    fun hleUtilsBufferCopyWithRange(output: SyncStream, input: SyncStream, command: CommandEnum): Unit {
        //println("hleUtilsBufferCopyWithRange(${input.length}): ${input.clone().readBytes(1024).hexString}")
        return when (command) {
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
    }

    fun kirk_CMD7(output: SyncStream, input: SyncStream) {
        output.clone().writeBytes(CMD7(input.sliceStart()))
    }

    fun kirk_CMD1(output: SyncStream, input: SyncStream) {
        //console.log(input.sliceWithLength(0, 128).readAllBytes());
        val header = input.sliceStart().read(AES128CMACHeader)
        if (header.Mode != KirkMode.Cmd1) throw invalidOp1("Kirk mode != Cmd1")
        val Keys = AES.decryptAes128Cbc(input.sliceStart().readBytes(32), KirkKeys.kirk1_key)
        val KeyAes = Keys.copyOfRange(0, 16)
        var KeyCmac = Keys.copyOfRange(16, 32)
        val PaddedDataSize = header.DataSize.nextAlignedTo(16)
        val PaddedData = ByteArray(PaddedDataSize)
        input.sliceStart()
            .also { it.skip(header.DataOffset + AES128CMACHeader.size) }
            .read(PaddedData)
        val Output = AES.decryptAes128Cbc(PaddedData, KeyAes)
        output.write(Output, 0, header.DataSize)
    }

    fun kirk_CMD17(input: SyncStream): Unit = TODO()
    fun kirk_CMD16(output: SyncStream, input: SyncStream): Unit = TODO()
    fun kirk_CMD14(output: SyncStream, outsize: Any): Unit = TODO()
    fun kirk_CMD13(output: SyncStream, input: SyncStream): Unit = TODO()
    fun kirk_CMD12(output: SyncStream): Unit = TODO()

    fun kirk_CMD11(output: SyncStream, input: SyncStream): Unit {
        if (input.length == 0L) invalidOp2
        val headerDataSize = input.sliceStart().readS32_le()
        if (headerDataSize == 0) invalidOp2
        val data = input.sliceStart(4L).readBytes(headerDataSize)
        val hash = data.hash(SHA1)
        output.writeBytes(hash.bytes)
    }

    fun kirk_CMD4(output: SyncStream, input: SyncStream): Unit = TODO()
    fun kirk_CMD10(input: SyncStream): Unit = TODO()

    fun CMD7(input: SyncStream): ByteArray {
        val header = input.read(KIRK_AES128CBC_HEADER)
        if (header.mode != KirkMode.DecryptCbc) {
            throw Error("Kirk Invalid mode '" + header.mode + "'")
        }
        if (header.data_size == 0) invalidOp1("Kirk data size == 0")
        return AES.decryptAes128Cbc(input.readAll(), getKirk7Key(header.keyseed))
        //return AES.decryptAes128Cbc(input.readBytes(header.data_size), kirk_4_7_get_key(header.keyseed))
    }

    fun getKirk7Key(key_type: Int): ByteArray = KirkKeys.kirk7_keys[key_type] ?: invalidOp1("Unsupported key $key_type")

    enum class KirkMode(override val id: Int) : IdEnum {
        Invalid0(0), Cmd1(1), Cmd2(2), Cmd3(3), EncryptCbc(4), DecryptCbc(5);

        companion object : UINT8_ENUM<KirkMode>(values())
    }

    enum class CommandEnum(override val id: Int) : IdEnum {
        DECRYPT_PRIVATE(0x1), ENCRYPT_SIGN(0x2), DECRYPT_SIGN(0x3), ENCRYPT_IV_0(0x4), ENCRYPT_IV_FUSE(0x5),
        ENCRYPT_IV_USER(0x6), DECRYPT_IV_0(0x7), DECRYPT_IV_FUSE(0x8), DECRYPT_IV_USER(0x9), PRIV_SIG_CHECK(0xA),
        SHA1_HASH(0xB), ECDSA_GEN_KEYS(0xC), ECDSA_MULTIPLY_POINT(0xD), PRNG(0xE), INIT(0xF), ECDSA_SIGN(0x10),
        ECDSA_VERIFY(0x11), CERT_VERIFY(0x12);

        companion object : UINT8_ENUM<CommandEnum>(values())
    }

    data class KIRK_AES128CBC_HEADER(
        var mode: KirkMode = KirkMode.Invalid0,
        var unk_4: Int = 0,
        var unk_8: Int = 0,
        var keyseed: Int = 0,
        var data_size: Int = 0
    ) {
        companion object : Struct<KIRK_AES128CBC_HEADER>(
            { KIRK_AES128CBC_HEADER() },
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
        companion object : Struct<AES128CMACHeader>(
            { AES128CMACHeader() },
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