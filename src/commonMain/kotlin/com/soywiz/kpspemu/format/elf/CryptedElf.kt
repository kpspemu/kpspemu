package com.soywiz.kpspemu.format.elf

import com.soywiz.kmem.*
import com.soywiz.korio.error.*
import com.soywiz.kpspemu.kirk.*
import com.soywiz.kpspemu.util.*
import com.soywiz.krypto.encoding.*

// https://github.com/hrydgard/ppsspp/blob/1f9fabee579422053d49e2de557aec6f20ee4405/Core/ELF/PrxDecrypter.cpp
// From PPSSPP
object CryptedElf {
    suspend fun decrypt(input: ByteArray): ByteArray {
        val out = ByteArray(input.size)
        val size = pspDecryptPRX(input.p_u8(), out.p_u8(), input.size)
        if (size < 0) invalidOp("Error decrypting prx")
        return out.copyOf(size)
    }

    private fun ROUNDUP16(x: Int) = (((x) + 15) and 15.inv())

    private fun GetTagInfo(checkTag: Int): Keys.TagInfo? = Keys.g_TagInfo.firstOrNull { it.tag == checkTag }

    private fun ExtraV2Mangle(buffer1: p_u8, codeExtra: Int) {
        val buffer2 = ByteArray(ROUNDUP16(0x14 + 0xA0)).p_u8()

        memcpy(buffer2 + 0x14, buffer1, 0xA0)

        val pl2 = buffer2.p_u32()
        pl2[0] = Kirk.KirkMode.DecryptCbc.id
        pl2[1] = 0
        pl2[2] = 0
        pl2[3] = codeExtra
        pl2[4] = 0xA0

        sceUtilsBufferCopyWithRange(buffer2, 20 + 0xA0, buffer2, 20 + 0xA0, KIRK_CMD_DECRYPT_IV_0)
        // copy result back
        memcpy(buffer1, buffer2, 0xA0)
    }

    private fun Scramble(buf: p_u32, size: Int, code: Int): Int {
        buf[0] = Kirk.KirkMode.DecryptCbc.id
        buf[1] = 0
        buf[2] = 0
        buf[3] = code
        buf[4] = size
        return sceUtilsBufferCopyWithRange(buf.p_u8(), size + 0x14, buf.p_u8(), size + 0x14, KIRK_CMD_DECRYPT_IV_0)
    }

    private fun DecryptPRX1(pbIn: p_u8, pbOut: p_u8, cbTotal: Int, tag: Int): Int {
        val bD0 = ByteArray(0x80).p_u8()
        val b80 = ByteArray(0x50).p_u8()
        val b00 = ByteArray(0x80).p_u8()
        val bB0 = ByteArray(0x20).p_u8()
        val pti = GetTagInfo(tag) ?: invalidOp("Missing tag ${tag.hex}")
        if (pti.key.size <= 0x10) return -1

        val firstZeroIndex = pti.key.data.indexOfFirst { it.toInt() != 0 }
        val retsize = ((pbIn + 0xB0).p_u32())[0]


        // Scramble the key (!)
        //
        // NOTE: I can't make much sense out of this code. Scramble seems really odd, appears
        // to write to stuff that should be before the actual key.
        //val key = ByteArray(0x14 + 0x90).p_u8()
        val key = ByteArray(0x90).p_u8()
        memcpy(key.p_u8(), pti.key.p_u8(), 0x90)
        if (firstZeroIndex < 0) {
            Scramble(key.p_u32(), 0x90, pti.code)
        }

        // build conversion into pbOut

        if (pbIn != pbOut) {
            memcpy(pbOut, pbIn, cbTotal)
        }

        memcpy(bD0, pbIn + 0xD0, 0x80)
        memcpy(b80, pbIn + 0x80, 0x50)
        memcpy(b00, pbIn + 0x00, 0x80)
        memcpy(bB0, pbIn + 0xB0, 0x20)

        memset(pbOut, 0, 0x150)
        memset(pbOut, 0x55, 0x40) // first $40 bytes ignored

        // step3 demangle in place
        val pl = (pbOut + 0x2C).p_u32()
        pl[0] = 5 // number of ulongs in the header
        pl[1] = 0
        pl[2] = 0
        pl[3] = pti.code // initial seed for PRX
        pl[4] = 0x70   // size

        // redo part of the SIG check (step2)
        val buffer1 = ByteArray(0x150).p_u8()
        memcpy(buffer1 + 0x00, bD0, 0x80)
        memcpy(buffer1 + 0x80, b80, 0x50)
        memcpy(buffer1 + 0xD0, b00, 0x80)
        if (pti.type != 0) {
            ExtraV2Mangle(buffer1 + 0x10, pti.type)
        }
        memcpy(pbOut + 0x40, buffer1 + 0x40, 0x40)

        for (iXOR in 0 until 0x70) {
            pbOut[0x40 + iXOR] = pbOut[0x40 + iXOR] xor key[0x14 + iXOR]
        }

        var ret = sceUtilsBufferCopyWithRange(pbOut + 0x2C, 20 + 0x70, pbOut + 0x2C, 20 + 0x70, KIRK_CMD_DECRYPT_IV_0)
        if (ret != 0) {
            invalidOp("Error(-1)")
        }

        for (iXOR in 0x6F downTo 0) { //for (iXOR in 0..0x6F) {
            pbOut[0x40 + iXOR] = pbOut[0x2C + iXOR] xor key[0x20 + iXOR]
        }

        memset(pbOut + 0x80, 0, 0x30) // $40 bytes kept, clean up
        pbOut[0xA0] = 1
        // copy unscrambled parts from header
        memcpy(pbOut + 0xB0, bB0, 0x20) // file size + lots of zeros
        memcpy(pbOut + 0xD0, b00, 0x80) // ~PSP header

        // step4: do the actual decryption of code block
        //  point 0x40 bytes into the buffer to key info
        ret = sceUtilsBufferCopyWithRange(pbOut, cbTotal, pbOut + 0x40, cbTotal - 0x40, KIRK_CMD_DECRYPT_PRIVATE)
        if (ret != 0) {
            invalidOp("Error(-2)")
        }

        // return cbTotal - 0x150; // rounded up size
        return retsize
    }

    private fun DecryptPRX2(inbuf: p_u8, outbuf: p_u8, size: Int, tag: Int): Int {
        val pti = GetTagInfo(tag) ?: return -1

        // only type2 and type6 can be process by this code.
        if (pti.type != 2 && pti.type != 6) {
            invalidOp("Error -12")
        }

        val retsize = (inbuf + 0xB0).p_u32()[0]
        val tmp1 = ByteArray(0x150).p_u8()
        val tmp2 = ByteArray(ROUNDUP16(0x90 + 0x14)).p_u8()
        val tmp3 = ByteArray(ROUNDUP16(0x90 + 0x14)).p_u8()
        val tmp4 = ByteArray(ROUNDUP16(0x20)).p_u8()

        if (inbuf != outbuf)
            memcpy(outbuf, inbuf, size)

        if (size < 0x160) {
            invalidOp("Error(-2)")
        }

        if ((size - 0x150) < retsize) {
            invalidOp("Error(-4)")
        }

        memcpy(tmp1, outbuf, 0x150)

        val p = tmp2 + 0x14

        // Writes 0x90 bytes to tmp2 + 0x14.
        for (i in 0 until 9) {
            memcpy(p + (i shl 4), pti.key.p_u8(), 0x10)
            p[(i shl 4)] = i   // really? this is very odd
        }

        if (Scramble(tmp2.p_u32(), 0x90, pti.code) < 0) {
            invalidOp("Error(-5) Scramble")
        }

        memcpy(outbuf, tmp1 + 0xD0, 0x5C)
        memcpy(outbuf + 0x5C, tmp1 + 0x140, 0x10)
        memcpy(outbuf + 0x6C, tmp1 + 0x12C, 0x14)
        memcpy(outbuf + 0x80, tmp1 + 0x080, 0x30)
        memcpy(outbuf + 0xB0, tmp1 + 0x0C0, 0x10)
        memcpy(outbuf + 0xC0, tmp1 + 0x0B0, 0x10)
        memcpy(outbuf + 0xD0, tmp1 + 0x000, 0x80)

        memcpy(tmp3 + 0x14, outbuf + 0x5C, 0x60)

        if (Scramble(tmp3.p_u32(), 0x60, pti.code) < 0) {
            invalidOp("Error(-6) Scramble")
        }

        memcpy(outbuf + 0x5C, tmp3, 0x60)
        memcpy(tmp3, outbuf + 0x6C, 0x14)
        memcpy(outbuf + 0x70, outbuf + 0x5C, 0x10)

        if (pti.type == 6) {
            memcpy(tmp4, outbuf + 0x3C, 0x20)
            memcpy(outbuf + 0x50, tmp4, 0x20)
            memset(outbuf + 0x18, 0, 0x38)
        } else
            memset(outbuf + 0x18, 0, 0x58)

        memcpy(outbuf + 0x04, outbuf, 0x04)
        outbuf.p_u32()[0] = 0x014C
        memcpy(outbuf + 0x08, tmp2, 0x10)

        /* sha-1 */

        if (sceUtilsBufferCopyWithRange(outbuf, 3000000, outbuf, 3000000, KIRK_CMD_SHA1_HASH) != 0) {
            invalidOp("Error(-7)")
        }

        if (memcmp(outbuf, tmp3, 0x14) != 0) {
            invalidOp("Error(-8)")
        }

        for (i in 0 until 0x40) {
            tmp3[i + 0x14] = outbuf[i + 0x80] xor tmp2[i + 0x10]
        }

        if (Scramble(tmp3.p_u32(), 0x40, pti.code) != 0) {
            invalidOp("Error(-9)")
        }

        for (i in 0 until 0x40) {
            outbuf[i + 0x40] = tmp3[i] xor tmp2[i + 0x50]
        }

        if (pti.type == 6) {
            memcpy(outbuf + 0x80, tmp4, 0x20)
            memset(outbuf + 0xA0, 0, 0x10)
            (outbuf + 0xA4).p_u32()[0] = 1
            (outbuf + 0xA0).p_u32()[0] = 1
        } else {
            memset(outbuf + 0x80, 0, 0x30)
            (outbuf + 0xA0).p_u32()[0] = 1
        }

        memcpy(outbuf + 0xB0, outbuf + 0xC0, 0x10)
        memset(outbuf + 0xC0, 0, 0x10)

        // The real decryption
        if (sceUtilsBufferCopyWithRange(outbuf, size, outbuf + 0x40, size - 0x40, KIRK_CMD_DECRYPT_PRIVATE) != 0) {
            invalidOp("Error(-1)")
        }

        if (retsize < 0x150) {
            // Fill with 0
            memset(outbuf + retsize, 0, 0x150 - retsize)
        }

        return retsize
    }

    private fun pspDecryptPRX(inbuf: p_u8, outbuf: p_u8, size: Int): Int {
        val retsize = DecryptPRX1(inbuf, outbuf, size, (inbuf + 0xD0).p_u32()[0])
        if (retsize >= 0 || retsize == MISSING_KEY) return retsize
        return DecryptPRX2(inbuf, outbuf, size, (inbuf + 0xD0).p_u32()[0])
    }

    private fun sceUtilsBufferCopyWithRange(
        output: p_u8,
        outputSize: Int,
        input: p_u8,
        inputSize: Int,
        command: Kirk.CommandEnum
    ): Int = Kirk.hleUtilsBufferCopyWithRange(output, outputSize, input, inputSize, command)

    private val KIRK_CMD_DECRYPT_PRIVATE = Kirk.CommandEnum.DECRYPT_PRIVATE // 0x1
    private val KIRK_CMD_DECRYPT_IV_0 = Kirk.CommandEnum.DECRYPT_IV_0 // 0x7
    private val KIRK_CMD_SHA1_HASH = Kirk.CommandEnum.SHA1_HASH // 0xB
    private val MISSING_KEY = -99

    @Suppress("RemoveRedundantCallsOfConversionMethods", "unused", "MemberVisibilityCanPrivate")
    object Keys {
        @Suppress("ArrayInDataClass")
        data class TagInfo(val tag: Int, val skey: ByteArray, val code: Int, val type: Int = 0) {
            val key = UByteArray(skey)
        }

        val g_TagInfo = listOf(
            TagInfo(
                tag = 0x4C949CF0.toInt(),
                skey = Hex.decode("3f6709a14771d69e277c7b32670e658a"),
                code = 0x43
            ), // keys210_vita_k0
            TagInfo(
                tag = 0x4C9494F0.toInt(),
                skey = Hex.decode("76f26c0aca3aba4eac76d240f5c3bff9"),
                code = 0x43
            ), // keys660_k1
            TagInfo(
                tag = 0x4C9495F0.toInt(),
                skey = Hex.decode("7a3e5575b96afc4f3ee3dfb36ce82a82"),
                code = 0x43
            ), // keys660_k2
            TagInfo(
                tag = 0x4C9490F0.toInt(),
                skey = Hex.decode("fa790936e619e8a4a94137188102e9b3"),
                code = 0x43
            ), // keys660_k3
            TagInfo(
                tag = 0x4C9491F0.toInt(),
                skey = Hex.decode("85931fed2c4da453599c3f16f350de46"),
                code = 0x43
            ), // keys660_k8
            TagInfo(
                tag = 0x4C9493F0.toInt(),
                skey = Hex.decode("c8a07098aee62b80d791e6ca4ca9784e"),
                code = 0x43
            ), // keys660_k4
            TagInfo(
                tag = 0x4C9497F0.toInt(),
                skey = Hex.decode("bff834028447bd871c52032379bb5981"),
                code = 0x43
            ), // keys660_k5
            TagInfo(
                tag = 0x4C9492F0.toInt(),
                skey = Hex.decode("d283cc63bb1015e77bc06dee349e4afa"),
                code = 0x43
            ), // keys660_k6
            TagInfo(
                tag = 0x4C9496F0.toInt(),
                skey = Hex.decode("ebd91e053caeab62e3b71f37e5cd68c3"),
                code = 0x43
            ), // keys660_k7
            TagInfo(
                tag = 0x457B90F0.toInt(),
                skey = Hex.decode("ba7661478b55a8728915796dd72f780e"),
                code = 0x5B
            ), // keys660_v1
            TagInfo(
                tag = 0x457B91F0.toInt(),
                skey = Hex.decode("c59c779c4101e48579c87163a57d4ffb"),
                code = 0x5B
            ), // keys660_v7
            TagInfo(
                tag = 0x457B92F0.toInt(),
                skey = Hex.decode("928ca412d65c55315b94239b62b3db47"),
                code = 0x5B
            ), // keys660_v6
            TagInfo(
                tag = 0x457B93F0.toInt(),
                skey = Hex.decode("88af18e9c3aa6b56f7c5a8bf1a84e9f3"),
                code = 0x5B
            ), // keys660_v3
            TagInfo(
                tag = 0x380290F0.toInt(),
                skey = Hex.decode("f94a6b96793fee0a04c88d7e5f383acf"),
                code = 0x5A
            ), // keys660_v2
            TagInfo(
                tag = 0x380291F0.toInt(),
                skey = Hex.decode("86a07d4db36ba2fdf41585702d6a0d3a"),
                code = 0x5A
            ), // keys660_v8
            TagInfo(
                tag = 0x380292F0.toInt(),
                skey = Hex.decode("d1b0aec324361349d649d788eaa49986"),
                code = 0x5A
            ), // keys660_v4
            TagInfo(
                tag = 0x380293F0.toInt(),
                skey = Hex.decode("cb93123831c02d2e7a185cac9293ab32"),
                code = 0x5A
            ), // keys660_v5
            TagInfo(
                tag = 0x4C948CF0.toInt(),
                skey = Hex.decode("017bf0e9be9add5437ea0ec4d64d8e9e"),
                code = 0x43
            ), // keys639_k3
            TagInfo(
                tag = 0x4C948DF0.toInt(),
                skey = Hex.decode("9843ff8568b2db3bd422d04fab5f0a31"),
                code = 0x43
            ), // keys638_k4
            TagInfo(
                tag = 0x4C948BF0.toInt(),
                skey = Hex.decode("91f2029e633230a91dda0ba8b741a3cc"),
                code = 0x43
            ), // keys636_k2
            TagInfo(
                tag = 0x4C948AF0.toInt(),
                skey = Hex.decode("07e308647f60a3366a762144c9d70683"),
                code = 0x43
            ), // keys636_k1
            TagInfo(
                tag = 0x457B8AF0.toInt(),
                skey = Hex.decode("47ec6015122ce3e04a226f319ffa973e"),
                code = 0x5B
            ), // keys636_1
            TagInfo(
                tag = 0x4C9487F0.toInt(),
                skey = Hex.decode("81d1128935c8ea8be0022d2d6a1867b8"),
                code = 0x43
            ), // keys630_k8
            TagInfo(
                tag = 0x457B83F0.toInt(),
                skey = Hex.decode("771c065f53ec3ffc22ce5a27ff78a848"),
                code = 0x5B
            ), // keys630_k7
            TagInfo(
                tag = 0x4C9486F0.toInt(),
                skey = Hex.decode("8ddbdc5cf2702b40b23d0009617c1060"),
                code = 0x43
            ), // keys630_k6
            TagInfo(
                tag = 0x457B82F0.toInt(),
                skey = Hex.decode("873721cc65aeaa5f40f66f2a86c7a1c8"),
                code = 0x5B
            ), // keys630_k5
            TagInfo(
                tag = 0x457B81F0.toInt(),
                skey = Hex.decode("aaa1b57c935a95bdef6916fc2b9231dd"),
                code = 0x5B
            ), // keys630_k4
            TagInfo(
                tag = 0x4C9485F0.toInt(),
                skey = Hex.decode("238d3dae4150a0faf32f32cec727cd50"),
                code = 0x43
            ), // keys630_k3
            TagInfo(
                tag = 0x457B80F0.toInt(),
                skey = Hex.decode("d43518022968fba06aa9a5ed78fd2e9d"),
                code = 0x5B
            ), // keys630_k2
            TagInfo(
                tag = 0x4C9484F0.toInt(),
                skey = Hex.decode("36b0dcfc592a951d802d803fcd30a01b"),
                code = 0x43
            ), // keys630_k1
            TagInfo(
                tag = 0x457B28F0.toInt(),
                skey = Hex.decode("b1b37f76c3fb88e6f860d3353ca34ef3"),
                code = 0x5B
            ), // keys620_e
            TagInfo(
                tag = 0x457B0CF0.toInt(),
                skey = Hex.decode("ac34bab1978dae6fbae8b1d6dfdff1a2"),
                code = 0x5B
            ), // keys620_a
            TagInfo(
                tag = 0x380228F0.toInt(),
                skey = Hex.decode("f28f75a73191ce9e75bd2726b4b40c32"),
                code = 0x5A
            ), // keys620_5v
            TagInfo(
                tag = 0x4C942AF0.toInt(),
                skey = Hex.decode("418a354f693adf04fd3946a25c2df221"),
                code = 0x43
            ), // keys620_5k
            TagInfo(
                tag = 0x4C9428F0.toInt(),
                skey = Hex.decode("f1bc1707aeb7c830d8349d406a8edf4e"),
                code = 0x43
            ), // keys620_5
            TagInfo(
                tag = 0x4C941DF0.toInt(),
                skey = Hex.decode("1d13e95004733dd2e1dab9c1e67b25a7"),
                code = 0x43
            ), // keys620_1
            TagInfo(
                tag = 0x4C941CF0.toInt(),
                skey = Hex.decode("d6bdce1e12af9ae66930deda88b8fffb"),
                code = 0x43
            ), // keys620_0
            TagInfo(
                tag = 0x4C9422F0.toInt(),
                skey = Hex.decode("e145932c53e2ab066fb68f0b6691e71e"),
                code = 0x43
            ), // keys600_2
            TagInfo(
                tag = 0x4C941EF0.toInt(),
                skey = Hex.decode("e35239973b84411cc323f1b8a9094bf0"),
                code = 0x43
            ), // keys600_1
            TagInfo(
                tag = 0x4C9429F0.toInt(),
                skey = Hex.decode("6d72a4ba7fbfd1f1a9f3bb071bc0b366"),
                code = 0x43
            ), // keys570_5k
            TagInfo(
                tag = 0x457B0BF0.toInt(),
                skey = Hex.decode("7b9472274ccc543baedf4637ac014d87"),
                code = 0x5B
            ), // keys505_a
            TagInfo(
                tag = 0x4C9419F0.toInt(),
                skey = Hex.decode("582a4c69197b833dd26161fe14eeaa11"),
                code = 0x43
            ), // keys505_1
            TagInfo(
                tag = 0x4C9418F0.toInt(),
                skey = Hex.decode("2e8e97a28542707318daa08af862a2b0"),
                code = 0x43
            ), // keys505_0
            TagInfo(
                tag = 0x457B1EF0.toInt(),
                skey = Hex.decode("a35d51e656c801cae377bfcdff24da4d"),
                code = 0x5B
            ), // keys500_c
            TagInfo(
                tag = 0x4C941FF0.toInt(),
                skey = Hex.decode("2c8eaf1dff79731aad96ab09ea35598b"),
                code = 0x43
            ), // keys500_2
            TagInfo(
                tag = 0x4C9417F0.toInt(),
                skey = Hex.decode("bae2a31207ff041b64a51185f72f995b"),
                code = 0x43
            ), // keys500_1
            TagInfo(
                tag = 0x4C9416F0.toInt(),
                skey = Hex.decode("eb1b530b624932581f830af4993d75d0"),
                code = 0x43
            ), // keys500_0
            TagInfo(
                tag = 0x4C9414F0.toInt(),
                skey = Hex.decode("45ef5c5ded81998412948fabe8056d7d"),
                code = 0x43
            ), // keys390_0
            TagInfo(
                tag = 0x4C9415F0.toInt(),
                skey = Hex.decode("701b082522a14d3b6921f9710aa841a9"),
                code = 0x43
            ), // keys390_1
            TagInfo(
                tag = 0x4C9412F0.toInt(),
                skey = Hex.decode("26380aaca5d874d132b72abf799e6ddb"),
                code = 0x43
            ), // keys370_0
            TagInfo(
                tag = 0x4C9413F0.toInt(),
                skey = Hex.decode("53e7abb9c64a4b779217b5740adaa9ea"),
                code = 0x43
            ), // keys370_1
            TagInfo(
                tag = 0x457B10F0.toInt(),
                skey = Hex.decode("7110f0a41614d59312ff7496df1fda89"),
                code = 0x5B
            ), // keys370_2
            TagInfo(
                tag = 0x4C940DF0.toInt(),
                skey = Hex.decode("3c2b51d42d8547da2dca18dffe5409ed"),
                code = 0x43
            ), // keys360_0
            TagInfo(
                tag = 0x4C9410F0.toInt(),
                skey = Hex.decode("311f98d57b58954532ab3ae389324b34"),
                code = 0x43
            ), // keys360_1
            TagInfo(
                tag = 0x4C940BF0.toInt(),
                skey = Hex.decode("3b9b1a56218014ed8e8b0842fa2cdc3a"),
                code = 0x43
            ), // keys330_0
            TagInfo(
                tag = 0x457B0AF0.toInt(),
                skey = Hex.decode("e8be2f06b1052ab9181803e3eb647d26"),
                code = 0x5B
            ), // keys330_1
            TagInfo(
                tag = 0x38020AF0.toInt(),
                skey = Hex.decode("ab8225d7436f6cc195c5f7f063733fe7"),
                code = 0x5A
            ), // keys330_2
            TagInfo(
                tag = 0x4C940AF0.toInt(),
                skey = Hex.decode("a8b14777dc496a6f384c4d96bd49ec9b"),
                code = 0x43
            ), // keys330_3
            TagInfo(
                tag = 0x4C940CF0.toInt(),
                skey = Hex.decode("ec3bd2c0fac1eeb99abcffa389f2601f"),
                code = 0x43
            ), // keys330_4
            TagInfo(
                tag = 0xCFEF09F0.toInt(),
                skey = Hex.decode("a241e839665bfabb1b2d6e0e33e5d73f"),
                code = 0x62
            ), // keys310_0
            TagInfo(
                tag = 0x457B08F0.toInt(),
                skey = Hex.decode("a4608fababdea5655d433ad15ec3ffea"),
                code = 0x5B
            ), // keys310_1
            TagInfo(
                tag = 0x380208F0.toInt(),
                skey = Hex.decode("e75c857a59b4e31dd09ecec2d6d4bd2b"),
                code = 0x5A
            ), // keys310_2
            TagInfo(
                tag = 0xCFEF08F0.toInt(),
                skey = Hex.decode("2e00f6f752cf955aa126b4849b58762f"),
                code = 0x62
            ), // keys310_3
            TagInfo(
                tag = 0xCFEF07F0.toInt(),
                skey = Hex.decode("7ba1e25a91b9d31377654ab7c28a10af"),
                code = 0x62
            ), // keys303_0
            TagInfo(
                tag = 0xCFEF06F0.toInt(),
                skey = Hex.decode("9f671a7a22f3590baa6da4c68bd00377"),
                code = 0x62
            ), // keys300_0
            TagInfo(
                tag = 0x457B06F0.toInt(),
                skey = Hex.decode("15076326dbe2693456082a934e4b8ab2"),
                code = 0x5B
            ), // keys300_1
            TagInfo(
                tag = 0x380206F0.toInt(),
                skey = Hex.decode("563b69f729882f4cdbd5de80c65cc873"),
                code = 0x5A
            ), // keys300_2
            TagInfo(
                tag = 0xCFEF05F0.toInt(),
                skey = Hex.decode("cafbbfc750eab4408e445c6353ce80b1"),
                code = 0x62
            ), // keys280_0
            TagInfo(
                tag = 0x457B05F0.toInt(),
                skey = Hex.decode("409bc69ba9fb847f7221d23696550974"),
                code = 0x5B
            ), // keys280_1
            TagInfo(
                tag = 0x380205F0.toInt(),
                skey = Hex.decode("03a7cc4a5b91c207fffc26251e424bb5"),
                code = 0x5A
            ), // keys280_2
            TagInfo(
                tag = 0x16D59E03.toInt(),
                skey = Hex.decode("c32489d38087b24e4cd749e49d1d34d1"),
                code = 0x62
            ), // keys260_0
            TagInfo(
                tag = 0x76202403.toInt(),
                skey = Hex.decode("f3ac6e7c040a23e70d33d82473392b4a"),
                code = 0x5B
            ), // keys260_1
            TagInfo(
                tag = 0x0F037303.toInt(),
                skey = Hex.decode("72b439ff349bae8230344a1da2d8b43c"),
                code = 0x5A
            ), // keys260_2
            TagInfo(
                tag = 0x4C940FF0.toInt(),
                skey = Hex.decode("8002c0bf000ac0bf4003c0bf40000000"),
                code = 0x43
            ), // key_2DA8
            TagInfo(
                tag = 0x4467415D.toInt(),
                skey = Hex.decode("660fcb3b3075e3100a9565c73c938722"),
                code = 0x59
            ), // key_22E0
            TagInfo(
                tag = 0x00000000.toInt(),
                skey = Hex.decode("6a1971f318ded3a26d3bdec7be98e24c"),
                code = 0x42
            ), // key_21C0
            TagInfo(
                tag = 0x01000000.toInt(),
                skey = Hex.decode("50cc03ac3f531afa0aa4342386617f97"),
                code = 0x43
            ), // key_2250
            TagInfo(
                tag = 0x2E5E10F0.toInt(),
                skey = Hex.decode("9d5c5baf8cd8697e519f7096e6d5c4e8"),
                code = 0x48
            ), // key_2E5E10F0
            TagInfo(
                tag = 0x2E5E12F0.toInt(),
                skey = Hex.decode("8a7bc9d6525888ea518360ca1679e207"),
                code = 0x48
            ), // key_2E5E12F0
            TagInfo(
                tag = 0x2E5E13F0.toInt(),
                skey = Hex.decode("ffa468c331cab74cf123ff01653d2636"),
                code = 0x48
            ), // key_2E5E13F0
            TagInfo(
                tag = 0x2FD30BF0.toInt(),
                skey = Hex.decode("d85879f9a422af8690acda45ce60403f"),
                code = 0x47
            ), // key_2FD30BF0
            TagInfo(
                tag = 0x2FD311F0.toInt(),
                skey = Hex.decode("3a6b489686a5c880696ce64bf6041744"),
                code = 0x47
            ), // key_2FD311F0
            TagInfo(
                tag = 0x2FD312F0.toInt(),
                skey = Hex.decode("c5fb6903207acfba2c90f8b84dd2f1de"),
                code = 0x47
            ), // key_2FD312F0
            TagInfo(
                tag = 0x2FD313F0.toInt(),
                skey = Hex.decode("b024c81643e8f01c8c3067733e9635ef"),
                code = 0x47
            ), // key_2FD313F0
            TagInfo(
                tag = 0xD91605F0.toInt(),
                skey = Hex.decode("b88c458bb6e76eb85159a6537c5e8631"),
                code = 0x5D,
                type = 2
            ), // key_D91605F0
            TagInfo(
                tag = 0xD91606F0.toInt(),
                skey = Hex.decode("ed10e036c4fe83f375705ef6a44005f7"),
                code = 0x5D,
                type = 2
            ), // key_D91606F0
            TagInfo(
                tag = 0xD91608F0.toInt(),
                skey = Hex.decode("5c770cbbb4c24fa27e3b4eb4b4c870af"),
                code = 0x5D,
                type = 2
            ), // key_D91608F0
            TagInfo(
                tag = 0xD91609F0.toInt(),
                skey = Hex.decode("d036127580562043c430943e1c75d1bf"),
                code = 0x5D,
                type = 2
            ), // key_D91609F0
            TagInfo(
                tag = 0xD9160AF0.toInt(),
                skey = Hex.decode("10a9ac16ae19c07e3b607786016ff263"),
                code = 0x5D,
                type = 2
            ), // key_D9160AF0
            TagInfo(
                tag = 0xD9160BF0.toInt(),
                skey = Hex.decode("8383f13753d0befc8da73252460ac2c2"),
                code = 0x5D,
                type = 2
            ), // key_D9160BF0
            TagInfo(
                tag = 0xD91611F0.toInt(),
                skey = Hex.decode("61b0c0587157d9fa74670e5c7e6e95b9"),
                code = 0x5D,
                type = 2
            ), // key_D91611F0
            TagInfo(
                tag = 0xD91612F0.toInt(),
                skey = Hex.decode("9e20e1cdd788dec0319b10afc5b87323"),
                code = 0x5D,
                type = 2
            ), // key_D91612F0
            TagInfo(
                tag = 0xD91613F0.toInt(),
                skey = Hex.decode("ebff40d8b41ae166913b8f64b6fcb712"),
                code = 0x5D,
                type = 2
            ), // key_D91613F0
            TagInfo(
                tag = 0xD91614F0.toInt(),
                skey = Hex.decode("fdf7b73c9fd1339511b8b5bb54237385"),
                code = 0x5D,
                type = 2
            ), // key_D91614F0
            TagInfo(
                tag = 0xD91615F0.toInt(),
                skey = Hex.decode("c803e34450f1e72a6a0dc361b68e5f51"),
                code = 0x5D,
                type = 2
            ), // key_D91615F0
            TagInfo(
                tag = 0xD91616F0.toInt(),
                skey = Hex.decode("5303b86a101998491caf30e4251b6b28"),
                code = 0x5D,
                type = 2
            ), // key_D91616F0
            TagInfo(
                tag = 0xD91617F0.toInt(),
                skey = Hex.decode("02fa487375afae0a67892b954b0987a3"),
                code = 0x5D,
                type = 2
            ), // key_D91617F0
            TagInfo(
                tag = 0xD91618F0.toInt(),
                skey = Hex.decode("96967cc3f712da621bf69a9a4444bc48"),
                code = 0x5D,
                type = 2
            ), // key_D91618F0
            TagInfo(
                tag = 0xD91619F0.toInt(),
                skey = Hex.decode("e032a7086b2b292cd14d5beea8c8b4e9"),
                code = 0x5D,
                type = 2
            ), // key_D91619F0
            TagInfo(
                tag = 0xD9161AF0.toInt(),
                skey = Hex.decode("27e5a74952e194673566910ce89a2524"),
                code = 0x5D,
                type = 2
            ), // key_D9161AF0
            TagInfo(
                tag = 0xD91620F0.toInt(),
                skey = Hex.decode("521cb45f403b9addacfcea92fdddf590"),
                code = 0x5D,
                type = 2
            ), // key_D91620F0
            TagInfo(
                tag = 0xD91621F0.toInt(),
                skey = Hex.decode("d1912ea621142962f6edaecbdda3bafe"),
                code = 0x5D,
                type = 2
            ), // key_D91621F0
            TagInfo(
                tag = 0xD91622F0.toInt(),
                skey = Hex.decode("595d784d21b201176c9ab51bdab7f9e6"),
                code = 0x5D,
                type = 2
            ), // key_D91622F0
            TagInfo(
                tag = 0xD91623F0.toInt(),
                skey = Hex.decode("aa45eb4f62fbd10d71d562d2f5bfa52f"),
                code = 0x5D,
                type = 2
            ), // key_D91623F0
            TagInfo(
                tag = 0xD91624F0.toInt(),
                skey = Hex.decode("61b726af8bf14158836ac49212cbb1e9"),
                code = 0x5D,
                type = 2
            ), // key_D91624F0
            TagInfo(
                tag = 0xD91628F0.toInt(),
                skey = Hex.decode("49a4fc66dce76221db18a750d6a8c1b6"),
                code = 0x5D,
                type = 2
            ), // key_D91628F0
            TagInfo(
                tag = 0xD91680F0.toInt(),
                skey = Hex.decode("2c229b123674116749d1d18892f6a1d8"),
                code = 0x5D,
                type = 6
            ), // key_D91680F0
            TagInfo(
                tag = 0xD91681F0.toInt(),
                skey = Hex.decode("52b6366c8c467f7acc116299c199be98"),
                code = 0x5D,
                type = 6
            ), // key_D91681F0
            TagInfo(
                tag = 0xD82310F0.toInt(),
                skey = Hex.decode("9d09fd20f38f10690db26f00ccc5512e"),
                code = 0x51
            ), // keys02G_E
            TagInfo(
                tag = 0xD8231EF0.toInt(),
                skey = Hex.decode("4f445c62b353c430fc3aa45becfe51ea"),
                code = 0x51
            ), // keys03G_E
            TagInfo(
                tag = 0xD82328F0.toInt(),
                skey = Hex.decode("5daa72f226604d1ce72dc8a32f79c554"),
                code = 0x51
            ), // keys05G_E
            TagInfo(
                tag = 0x279D08F0.toInt(),
                skey = Hex.decode("c7277285aba7f7f04cc186cce37f17ca"),
                code = 0x61
            ), // oneseg_310
            TagInfo(
                tag = 0x279D06F0.toInt(),
                skey = Hex.decode("76409e08db9b3ba1478a968ef3f76292"),
                code = 0x61
            ), // oneseg_300
            TagInfo(
                tag = 0x279D05F0.toInt(),
                skey = Hex.decode("23dc3bb5a982d6ea63a36e2b2be9e154"),
                code = 0x61
            ), // oneseg_280
            TagInfo(
                tag = 0xD66DF703.toInt(),
                skey = Hex.decode("224357682f41ce654ca37cc6c4acf360"),
                code = 0x61
            ), // oneseg_260_271
            TagInfo(
                tag = 0x279D10F0.toInt(),
                skey = Hex.decode("12570d8a166d8706037dc88b62a332a9"),
                code = 0x61
            ), // oneseg_slim
            TagInfo(
                tag = 0x3C2A08F0.toInt(),
                skey = Hex.decode("1e2e3849dad41608272ef3bc37758093"),
                code = 0x67
            ), // ms_app_main
            TagInfo(
                tag = 0xADF305F0.toInt(),
                skey = Hex.decode("1299705e24076cd02d06fe7eb30c1126"),
                code = 0x60
            ), // demokeys_280
            TagInfo(
                tag = 0xADF306F0.toInt(),
                skey = Hex.decode("4705d5e3561e819b092f06db6b1292e0"),
                code = 0x60
            ), // demokeys_3XX_1
            TagInfo(
                tag = 0xADF308F0.toInt(),
                skey = Hex.decode("f662396e26224dca026416997b9ae7b8"),
                code = 0x60
            ), // demokeys_3XX_2
            TagInfo(
                tag = 0x8004FD03.toInt(),
                skey = Hex.decode("f4aef4e186ddd29c7cc542a695a08388"),
                code = 0x5D,
                type = 2
            ), // ebootbin_271_new
            TagInfo(
                tag = 0xD91605F0.toInt(),
                skey = Hex.decode("b88c458bb6e76eb85159a6537c5e8631"),
                code = 0x5D
            ), // ebootbin_280_new
            TagInfo(
                tag = 0xD91606F0.toInt(),
                skey = Hex.decode("ed10e036c4fe83f375705ef6a44005f7"),
                code = 0x5D
            ), // ebootbin_300_new
            TagInfo(
                tag = 0xD91608F0.toInt(),
                skey = Hex.decode("5c770cbbb4c24fa27e3b4eb4b4c870af"),
                code = 0x5D
            ), // ebootbin_310_new
            TagInfo(
                tag = 0x0A35EA03.toInt(),
                skey = Hex.decode("f948380c9688a7744f65a054c276d9b8"),
                code = 0x5E
            ), // gameshare_260_271
            TagInfo(
                tag = 0x7B0505F0.toInt(),
                skey = Hex.decode("2d86773a56a44fdd3c167193aa8e1143"),
                code = 0x5E
            ), // gameshare_280
            TagInfo(
                tag = 0x7B0506F0.toInt(),
                skey = Hex.decode("781ad28724bda296183f893672909285"),
                code = 0x5E
            ), // gameshare_300
            TagInfo(
                tag = 0x7B0508F0.toInt(),
                skey = Hex.decode("c97d3e0a54816ec7137499746218e7dd"),
                code = 0x5E
            ), // gameshare_310
            TagInfo(
                tag = 0x380210F0.toInt(),
                skey = Hex.decode("322cfa75e47e93eb9f22808557089848"),
                code = 0x5A
            ), // key_380210F0
            TagInfo(
                tag = 0x380280F0.toInt(),
                skey = Hex.decode("970912d3db02bdd8e77451fef0ea6c5c"),
                code = 0x5A
            ), // key_380280F0
            TagInfo(
                tag = 0x380283F0.toInt(),
                skey = Hex.decode("34200c8ea1867984af13ae34776fea89"),
                code = 0x5A
            ), // key_380283F0
            TagInfo(
                tag = 0x407810F0.toInt(),
                skey = Hex.decode("afadcaf1955991ec1b27d04e8af33de7"),
                code = 0x6A
            ), // key_407810F0
            TagInfo(
                tag = 0xE92410F0.toInt(),
                skey = Hex.decode("36ef824e74fb175b141405f3b38a7618"),
                code = 0x40
            ), // drmkeys_6XX_1
            TagInfo(
                tag = 0x692810F0.toInt(),
                skey = Hex.decode("21525d76f6810f152f4a408963a01055"),
                code = 0x40
            ), // drmkeys_6XX_2
            TagInfo(
                tag = 0x00000000.toInt(),
                skey = Hex.decode("bef3217b1d5e9c29715e9c1c4546cb96e01b9b3c3dde85eb22207f4aaa6e20c265320bd5670577554008083cf2551d98f3f6d85fc5b08eee52814d94518627f8faba052733e52084e94a152732aa194840aaa35965cfb32c6d4674f20556653a8ff8b021268db1c55190c1644ec969d6f23570e809593a9d02714e6fce46a9dc1b881684a597b0bac625912472084cb3"),
                code = 0x42,
                type = 0x00
            ), // g_key0
            TagInfo(
                tag = 0x02000000.toInt(),
                skey = Hex.decode("32a9fdcc766fc051cfcc6d041e82e1494c023b7d6558da9d25988cccb57de9d1cbd8746887c97134fcb3ed725d36c8813ae361e159db92fcecb10920e44ca9b16b69032fd836e287e98c2b3b84e70503830871f939db39b037ea3b8905684de7bd385c2a13c88db07523b3152545be4690fd0301a2870ea96aa6ab52807bbf8563cee845d316d74d2d0de3f556e43aaf"),
                code = 0x45,
                type = 0x00
            ), // g_key2
            TagInfo(
                tag = 0x03000000.toInt(),
                skey = Hex.decode("caf5c8a680c0676d3a4d4f926aa07c049702640858a7d44f875a68bdc201279b352ab6833c536b720cfa22e5b4064bc2ac1c9d457b41c5a8a262ea4f42d71506098d623014ab4fc45e71ff697d83d8d28b0bedbeae576e1e02c4e861067a36be5e2b3f5458c03edb752085becc4d7e1e55ea6415b42578ecad8c53c07f2cf770d0c3e849c57ea9eda4b092f42ab05ee0"),
                code = 0x46,
                type = 0x00
            ), // g_key3
            TagInfo(
                tag = 0x4467415D.toInt(),
                skey = Hex.decode("05e080ef9f68543acd9cc943be27771b3800b85c62fe2edd2cf969f3c5940f1619005629c5103cbf6655cef226c6a2ce6f8101b61e48e764bdde340cb09cf298d704c53ff039fbc8d8b32102a236f96300483a9ae332cc6efd0c128e231636b089e6e1aeeb0255741cc6a6e4b43ef2741358fad7eb1619b057843212d297bcd2d8256464a5808332b18ada43c92a124b"),
                code = 0x59,
                type = 0x59
            ), // g_key44
            TagInfo(
                tag = 0x207BBF2F.toInt(),
                skey = Hex.decode("0008b533cd5f2ff31f88143c952a8a6ed5effe29e3ea941343d46bbd83c02108d379b3fa65e113e6d354a7f552298b10151e4b0abadeea61df6575550153463bc3ec54ae0933426119ffc970ece50a5b26f19d985f7a989d0e75bc5527ba6ec6e888e92dda0066f7cbdc8203f2f569556212438ed3e38f2887216f659c2ed137b49e532f8e9992a4f75839ed2365e939"),
                code = 0x5A,
                type = 0x5A
            ), // g_key20
            TagInfo(
                tag = 0x3ACE4DCE.toInt(),
                skey = Hex.decode("697087671756bd3adcb13ac27d5057ab407f6a06b9f9de24e459f706b124f5dc5e3e79132d025903a2e1e7aafab2b9764003169aba2f8287bb8fe219028a339e9a7e00d8f17a31eade7106637cca670baf92518626353cea8e8c442b5492598bcbe90246da6ce14dbbd564e18ed8ec07f8e5ff99c1008876ed91b05334740484bcdb26b4bb48f9365821144692b49b74"),
                code = 0x5B,
                type = 0x5B
            ), // g_key3A
            TagInfo(
                tag = 0x07000000.toInt(),
                skey = Hex.decode("af2f36f96a73820ead3594e9e9870c89bbe84f57ad3436db805d48fe5cb5e9e95b46b5ac"),
                code = 0x4A,
                type = 0x00
            ), // g_key_INDEXDAT1xx
            TagInfo(
                tag = 0x08000000.toInt(),
                skey = Hex.decode("ef69cb1812898e15bb0ef9de23fbb04c18ee87366e4a8d8656c7b5191d5516ee6c2dcbe760c647973f1495ce77f45629de4a8203f19d0c2124eb29509fe6df81009bc839918b0cb0c2f92deffc933ae1a8a4948b9dd01d490d406a68e4c7d4cec9b7c89628dcaa1e840b17a4dc5d5d50cfc3a65d2dfa5d0eb519796ec7295ece94dbacaadd0cf7452537a7623d56e6cc"),
                code = 0x4B,
                type = 0x00
            ), // g_keyEBOOT1xx
            TagInfo(
                tag = 0xC0CB167C.toInt(),
                skey = Hex.decode("fa368eda4774d95d7498c176af7ee597bd09ab1cc6ba35988192d303cf05b20334e7822863f614e775276eb9c7af8abd29ecd31d6ca1a4ec87ec695f921e988521aefc7c16dde9ba0478a9e6fc02ee2e3d8adf61640531dd49e197963b3f45c256841df9c86bda39f5fee5b3a393c589bc8a5cfb12720b6ccbd30de1a8b2d0984718d65f5723dcf06a160177683b5c0f"),
                code = 0x5D,
                type = 0x5D
            ), // g_keyEBOOT2xx
            TagInfo(
                tag = 0x0B000000.toInt(),
                skey = Hex.decode("bf3c60a5412448d7cc6457f60b06901f453ea74e92d151e58a5db7e76e505a462210fb405033272c44da96808e19479977ee8d272e065d7445fa48c1af822583da86db5fcec415cb2fc62425b1c32e6c9ee39b36c41febf71ace511ef43605d7d8394dc313fb1874e14dc8e33cf018b14e8d01a20d77d8e690f320574163f9178fa6a46028dd2713644c9405124c2c0c"),
                code = 0x4E,
                type = 0x00
            ), // g_keyUPDATER
            TagInfo(
                tag = 0x0C000000.toInt(),
                skey = Hex.decode("2f10bf1a71d096d5b252c56f1f53f2d4d9cd25f003af9aafcf57cfe0c49454255e67037084c87b90e44e2d000d7a680b4fa43a9e81da8ff58cac26ec9db4c93a37c071344d83f3b01144dc1031ea32a26bfae5e2034b5945871c3ae4d1d9da310370cd08df2f9cfa251d895a34195c9be566f322324a085fd51655699fbe452205d76d4fa1b8b8c400a613bc3bfcb777"),
                code = 0x4F,
                type = 0x00
            ), // g_keyDEMOS27X
            TagInfo(
                tag = 0x0F000000.toInt(),
                skey = Hex.decode("bcfe81a3c9d5b9998d0a566c959f3030cc4626795e4eb682ad51391ac42e180ab43161c48a0cc577c6165f322e94d102c48aa30ac60a942a2647036733b12de50721efd2901ec885ba64d1c81dce8dc375a28b940346b80d373647e2dafc74cd663d8e5822e8286d8b541e896df53cf566dbbd0baa86b2c44bbceb2bf41f26fc05e7b8925269eedce542045e217feb8b"),
                code = 0x52,
                type = 0x00
            ), // g_keyMEIMG250
            TagInfo(
                tag = 0x862648D1.toInt(),
                skey = Hex.decode("98d6bf1124b3f9d7274952dd865b21166dc34a5017b2435847daa0e5e7a173bb35db15293afd5c3705a970bbcaef2b279107962ebb9907eac8e65ab873f7cac941e60e259e4ae7065d894452a555674653af849a744102e11e03baeeceb980ed725f31bc7f062158583031e806e7d0d23e93d8e6b47fd1d7c49650503b0ba5fd3dae35468a9c48eb2d762d4231328b5a"),
                code = 0x52,
                type = 0x52
            ), // g_keyMEIMG260
            TagInfo(
                tag = 0x207BBF2F.toInt(),
                skey = Hex.decode("0008b533cd5f2ff31f88143c952a8a6ed5effe29e3ea941343d46bbd83c02108d379b3fa65e113e6d354a7f552298b10151e4b0abadeea61df6575550153463bc3ec54ae0933426119ffc970ece50a5b26f19d985f7a989d0e75bc5527ba6ec6e888e92dda0066f7cbdc8203f2f569556212438ed3e38f2887216f659c2ed137b49e532f8e9992a4f75839ed2365e939"),
                code = 0x5A,
                type = 0x5A
            ), // g_keyUNK1
            TagInfo(
                tag = 0x09000000.toInt(),
                skey = Hex.decode("e8531b72c6313efca2a25bf872acf03caba7ee54cbbf59596b83b854131343bccff29e98b236cef0f84cba9831c971e9c85d37a0a02fe50826d40dac01d6e457c7616ec58ab91aeff4f8d9d108a7e95f079df03e8c1a0cfa5cea1ea9c582f4580203802cc3f6e67ebbbb6affd0d01021887a29d3d31200987bc859dc9257dc7fa65d3fdb87b723fcd38e692212e880b6"),
                code = 0x4C,
                type = 0x00
            ), // g_key_GAMESHARE1xx
            TagInfo(
                tag = 0xBB67C59F.toInt(),
                skey = Hex.decode("c757a7943398d39f718350f8290b8b32dab9bc2cc6b91829ba504c94d0e7dcf166390c64083d0bc9ba17adf44bf8a06c677c76f75aa5d3a46a5c084a7170b26bfb388bfab831db3ff296718b4aed9bdb845b6251b481144c08f584f67047b430748eaa93bc79c5908dc86e240212052e2e8474c797d985a1dd3a2b7a6d5b83fe4d188f50134f4cebd393190ed2df96ba"),
                code = 0x5E,
                type = 0x5E
            )  // g_key_GAMESHARE2xx
        )
    }

    //@Suppress("ArrayInDataClass")
    //data class Header(
    //	var magic: Int = 0,
    //	var modAttr: Int = 0,
    //	var compModAttr: Int = 0,
    //	var modVerLo: Int = 0,
    //	var modVerHi: Int = 0,
    //	var moduleName: String = "",
    //	var modVersion: Int = 0,
    //	var nsegments: Int = 0,
    //	var elfSize: Int = 0,
    //	var pspSize: Int = 0,
    //	var bootEntry: Int = 0,
    //	var modInfoOffset: Int = 0,
    //	var bssSize: Int = 0,
    //	var segAlign: CharArray = CharArray(4),
    //	var segAddress: IntArray = IntArray(4),
    //	var segSize: IntArray = IntArray(4),
    //	var reserved: IntArray = IntArray(5),
    //	var devkitVersion: Int = 0,
    //	var decMode: Int = 0,
    //	var pad: Int = 0,
    //	var overlapSize: Int = 0,
    //	var aesKey: ByteArray = ByteArray(16),
    //	var cmacKey: ByteArray = ByteArray(16),
    //	var cmacHeaderHash: ByteArray = ByteArray(16),
    //	var compressedSize: Int = 0,
    //	var compressedOffset: Int = 0,
    //	var unk1: Int = 0,
    //	var unk2: Int = 0,
    //	var cmacDataHash: ByteArray = ByteArray(16),
    //	var tag: Int = 0,
    //	var sigcheck: ByteArray = ByteArray(88),
    //	var sha1Hash: ByteArray = ByteArray(20),
    //	var keyData: ByteArray = ByteArray(16)
    //) {
    //	companion object : Struct<Header>({ Header() },
    //		Header::magic AS INT32,                   // 0000
    //		Header::modAttr AS UINT16,                // 0004
    //		Header::compModAttr AS UINT16,            // 0006
    //		Header::modVerLo AS UINT8,                // 0008
    //		Header::modVerHi AS UINT8,                // 0009
    //		Header::moduleName AS STRINGZ(28),        // 000A
    //		Header::modVersion AS UINT8,              // 0026
    //		Header::nsegments AS UINT8,               // 0027
    //		Header::elfSize AS INT32,                 // 0028
    //		Header::pspSize AS INT32,                 // 002C
    //		Header::bootEntry AS INT32,               // 0030
    //		Header::modInfoOffset AS INT32,           // 0034
    //		Header::bssSize AS INT32,                 // 0038
    //		Header::segAlign AS CHARARRAY(4),         // 003C
    //		Header::segAddress AS INTARRAY(4),        // 0044
    //		Header::segSize AS INTARRAY(4),           // 0054
    //		Header::reserved AS INTARRAY(5),          // 0064
    //		Header::devkitVersion AS INT32,           // 0078
    //		Header::decMode AS UINT8,                 // 007C
    //		Header::pad AS UINT8,                     // 007D
    //		Header::overlapSize AS UINT16,            // 007E
    //		Header::aesKey AS BYTEARRAY(16),          // 0080
    //		Header::cmacKey AS BYTEARRAY(16),         // 0090
    //		Header::cmacHeaderHash AS BYTEARRAY(16),  // 00A0
    //		Header::compressedSize AS INT32,          // 00B0
    //		Header::compressedOffset AS INT32,        // 00B4
    //		Header::unk1 AS INT32,                    // 00B8
    //		Header::unk2 AS INT32,                    // 00BC
    //		Header::cmacDataHash AS BYTEARRAY(16),    // 00D0
    //		Header::tag AS INT32,                     // 00E0
    //		Header::sigcheck AS BYTEARRAY(88),        // 00E4
    //		Header::sha1Hash AS BYTEARRAY(20),        // 013C
    //		Header::keyData AS BYTEARRAY(16)          // 0150-0160
    //	)
    //}
}

//fun main(args: Array<String>) = Korio {
//	val ebootBin = LocalVfs("c:/temp/1/EBOOT.BIN").readAll()
//	//val decrypted1 = CryptedElf.decrypt(ebootBin)
//	val decrypted2 = CryptedElf.decrypt(ebootBin)
//	//LocalVfs("c:/temp/1/EBOOT.BIN.decrypted.kpspemu1").writeBytes(decrypted1)
//	LocalVfs("c:/temp/1/EBOOT.BIN.decrypted.kpspemu2").writeBytes(decrypted2)
//}
