package com.soywiz.kpspemu.kirk

import com.soywiz.krypto.encoding.*

@Suppress("unused")
object KirkKeys {
    var kirk1_key = Hex.decode("98C940975C1D10E87FE60EA3FD03A8BA")
    var kirk16_key = Hex.decode("475E09F4A237DA9BEFFF3BC077143D8A")

    var kirk7_keys = hashMapOf(
        0x02 to Hex.decode("B813C35EC64441E3DC3C16F5B45E6484"), // New from PS3
        0x03 to Hex.decode("9802C4E6EC9E9E2FFC634CE42FBB4668"),
        0x04 to Hex.decode("99244CD258F51BCBB0619CA73830075F"),
        0x05 to Hex.decode("0225D7BA63ECB94A9D237601B3F6AC17"),
        0x07 to Hex.decode("76368B438F77D87EFE5FB6115939885C"), // New from PS3
        0x0C to Hex.decode("8485C848750843BC9B9AECA79C7F6018"),
        0x0D to Hex.decode("B5B16EDE23A97B0EA17CDBA2DCDEC46E"),
        0x0E to Hex.decode("C871FDB3BCC5D2F2E2D7729DDF826882"),
        0x0F to Hex.decode("0ABB336C96D4CDD8CB5F4BE0BADB9E03"),
        0x10 to Hex.decode("32295BD5EAF7A34216C88E48FF50D371"),
        0x11 to Hex.decode("46F25E8E4D2AA540730BC46E47EE6F0A"),
        0x12 to Hex.decode("5DC71139D01938BC027FDDDCB0837D9D"),
        0x38 to Hex.decode("12468D7E1C42209BBA5426835EB03303"),
        0x39 to Hex.decode("C43BB6D653EE67493EA95FBC0CED6F8A"),
        0x3A to Hex.decode("2CC3CF8C2878A5A663E2AF2D715E86BA"),
        0x44 to Hex.decode("7DF49265E3FAD678D6FE78ADBB3DFB63"), // New from PS3
        0x4B to Hex.decode("0CFD679AF9B4724FD78DD6E99642288B"), // 1.xx game eboot.bin
        0x53 to Hex.decode("AFFE8EB13DD17ED80A61241C959256B6"),
        0x57 to Hex.decode("1C9BC490E3066481FA59FDB600BB2870"),
        0x5D to Hex.decode("115A5D20D53A8DD39CC5AF410F0F186F"),
        0x63 to Hex.decode("9C9B1372F8C640CF1C62F5D592DDB582"),
        0x64 to Hex.decode("03B302E85FF381B13B8DAA2A90FF5E61")
    )

    // ECC Curves for Kirk 1 and Kirk 0x11
    // Common Curve paramters p and a
    var ec_p = Hex.decode("FFFFFFFFFFFFFFFF00000001FFFFFFFFFFFFFFFF")
    var ec_a = Hex.decode("FFFFFFFFFFFFFFFF00000001FFFFFFFFFFFFFFFC") // mon

    // Kirk 0xC,0xD,0x10,0x11,(likely 0x12)- Unique curve parameters for b, N, and base point G for Kirk 0xC,0xD,0x10,0x11,(likely 0x12) service
    // Since public key is variable, it is not specified here
    var ec_b2 = Hex.decode("A68BEDC33418029C1D3CE33B9A321FCCBB9E0F0B")
    var ec_N2 = Hex.decode("00FFFFFFFFFFFFFFFEFFFFB5AE3C523E63944F2127")
    var Gx2 = Hex.decode("128EC4256487FD8FDF64E2437BC0A1F6D5AFDE2C")
    var Gy2 = Hex.decode("5958557EB1DB001260425524DBC379D5AC5F4ADF")

    // KIRK 1 - Unique curve parameters for b, N, and base point G
    // Since public key is hard coded, it is also included

    var ec_b1 = Hex.decode("65D1488C0359E234ADC95BD3908014BD91A525F9")
    var ec_N1 = Hex.decode("00FFFFFFFFFFFFFFFF0001B5C617F290EAE1DBAD8F")
    var Gx1 = Hex.decode("2259ACEE15489CB096A882F0AE1CF9FD8EE5F8FA")
    var Gy1 = Hex.decode("604358456D0A1CB2908DE90F27D75C82BEC108C0")
    var Px1 = Hex.decode("ED9CE58234E61A53C685D64D51D0236BC3B5D4B9")
    var Py1 = Hex.decode("049DF1A075C0E04FB344858B61B79B69A63D2C39 ")
}