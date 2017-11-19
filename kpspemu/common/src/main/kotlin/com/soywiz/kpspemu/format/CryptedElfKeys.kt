package com.soywiz.kpspemu.format

import com.soywiz.kmem.UByteArray
import com.soywiz.korio.crypto.Hex

@Suppress("RemoveRedundantCallsOfConversionMethods", "MemberVisibilityCanPrivate", "unused")
object CryptedPrxKeys144 {
	@Suppress("ArrayInDataClass")
	data class TagInfo(val tag: Int, val skey: ByteArray, val code: Int, val codeExtra: Int) {
		val key: UByteArray = UByteArray(skey)
	}

	val g_tagInfo = listOf(
		TagInfo(tag = 0x00000000.toInt(), skey = Hex.decode("bef3217b1d5e9c29715e9c1c4546cb96e01b9b3c3dde85eb22207f4aaa6e20c265320bd5670577554008083cf2551d98f3f6d85fc5b08eee52814d94518627f8faba052733e52084e94a152732aa194840aaa35965cfb32c6d4674f20556653a8ff8b021268db1c55190c1644ec969d6f23570e809593a9d02714e6fce46a9dc1b881684a597b0bac625912472084cb3"), code = 0x00000042, codeExtra = 0x00000000),
		TagInfo(tag = 0x02000000.toInt(), skey = Hex.decode("32a9fdcc766fc051cfcc6d041e82e1494c023b7d6558da9d25988cccb57de9d1cbd8746887c97134fcb3ed725d36c8813ae361e159db92fcecb10920e44ca9b16b69032fd836e287e98c2b3b84e70503830871f939db39b037ea3b8905684de7bd385c2a13c88db07523b3152545be4690fd0301a2870ea96aa6ab52807bbf8563cee845d316d74d2d0de3f556e43aaf"), code = 0x00000045, codeExtra = 0x00000000),
		TagInfo(tag = 0x03000000.toInt(), skey = Hex.decode("caf5c8a680c0676d3a4d4f926aa07c049702640858a7d44f875a68bdc201279b352ab6833c536b720cfa22e5b4064bc2ac1c9d457b41c5a8a262ea4f42d71506098d623014ab4fc45e71ff697d83d8d28b0bedbeae576e1e02c4e861067a36be5e2b3f5458c03edb752085becc4d7e1e55ea6415b42578ecad8c53c07f2cf770d0c3e849c57ea9eda4b092f42ab05ee0"), code = 0x00000046, codeExtra = 0x00000000),
		TagInfo(tag = 0x4467415D.toInt(), skey = Hex.decode("05e080ef9f68543acd9cc943be27771b3800b85c62fe2edd2cf969f3c5940f1619005629c5103cbf6655cef226c6a2ce6f8101b61e48e764bdde340cb09cf298d704c53ff039fbc8d8b32102a236f96300483a9ae332cc6efd0c128e231636b089e6e1aeeb0255741cc6a6e4b43ef2741358fad7eb1619b057843212d297bcd2d8256464a5808332b18ada43c92a124b"), code = 0x00000059, codeExtra = 0x00000059),
		TagInfo(tag = 0x207BBF2F.toInt(), skey = Hex.decode("0008b533cd5f2ff31f88143c952a8a6ed5effe29e3ea941343d46bbd83c02108d379b3fa65e113e6d354a7f552298b10151e4b0abadeea61df6575550153463bc3ec54ae0933426119ffc970ece50a5b26f19d985f7a989d0e75bc5527ba6ec6e888e92dda0066f7cbdc8203f2f569556212438ed3e38f2887216f659c2ed137b49e532f8e9992a4f75839ed2365e939"), code = 0x0000005A, codeExtra = 0x0000005A),
		TagInfo(tag = 0x3ACE4DCE.toInt(), skey = Hex.decode("697087671756bd3adcb13ac27d5057ab407f6a06b9f9de24e459f706b124f5dc5e3e79132d025903a2e1e7aafab2b9764003169aba2f8287bb8fe219028a339e9a7e00d8f17a31eade7106637cca670baf92518626353cea8e8c442b5492598bcbe90246da6ce14dbbd564e18ed8ec07f8e5ff99c1008876ed91b05334740484bcdb26b4bb48f9365821144692b49b74"), code = 0x0000005B, codeExtra = 0x0000005B),
		TagInfo(tag = 0x07000000.toInt(), skey = Hex.decode("af00cb762fe61c11367eb2b7f9e88d6d6af14bd57303e9d982d999750e2bf851ad03616335bc408e942c332fe9aa13f5e9fe2ad2873934040cb85bfc899d3412bb81a414e83aed254f0e507d57b7d143adfd597b34bffb4c3674d1c3db21dac1808c4da35d232b964805423efe9fcf095c3f88d4b59c0ed9e9f4ae00e96d88f05b8aa5624655a552b5411997ac9fb7f5"), code = 0x0000004A, codeExtra = 0x00000000),
		TagInfo(tag = 0x08000000.toInt(), skey = Hex.decode("ef69cb1812898e15bb0ef9de23fbb04c18ee87366e4a8d8656c7b5191d5516ee6c2dcbe760c647973f1495ce77f45629de4a8203f19d0c2124eb29509fe6df81009bc839918b0cb0c2f92deffc933ae1a8a4948b9dd01d490d406a68e4c7d4cec9b7c89628dcaa1e840b17a4dc5d5d50cfc3a65d2dfa5d0eb519796ec7295ece94dbacaadd0cf7452537a7623d56e6cc"), code = 0x0000004B, codeExtra = 0x00000000),
		TagInfo(tag = 0xC0CB167C.toInt(), skey = Hex.decode("fa368eda4774d95d7498c176af7ee597bd09ab1cc6ba35988192d303cf05b20334e7822863f614e775276eb9c7af8abd29ecd31d6ca1a4ec87ec695f921e988521aefc7c16dde9ba0478a9e6fc02ee2e3d8adf61640531dd49e197963b3f45c256841df9c86bda39f5fee5b3a393c589bc8a5cfb12720b6ccbd30de1a8b2d0984718d65f5723dcf06a160177683b5c0f"), code = 0x0000005D, codeExtra = 0x0000005D),
		TagInfo(tag = 0x0B000000.toInt(), skey = Hex.decode("bf3c60a5412448d7cc6457f60b06901f453ea74e92d151e58a5db7e76e505a462210fb405033272c44da96808e19479977ee8d272e065d7445fa48c1af822583da86db5fcec415cb2fc62425b1c32e6c9ee39b36c41febf71ace511ef43605d7d8394dc313fb1874e14dc8e33cf018b14e8d01a20d77d8e690f320574163f9178fa6a46028dd2713644c9405124c2c0c"), code = 0x0000004E, codeExtra = 0x00000000),
		TagInfo(tag = 0x0C000000.toInt(), skey = Hex.decode("2f10bf1a71d096d5b252c56f1f53f2d4d9cd25f003af9aafcf57cfe0c49454255e67037084c87b90e44e2d000d7a680b4fa43a9e81da8ff58cac26ec9db4c93a37c071344d83f3b01144dc1031ea32a26bfae5e2034b5945871c3ae4d1d9da310370cd08df2f9cfa251d895a34195c9be566f322324a085fd51655699fbe452205d76d4fa1b8b8c400a613bc3bfcb777"), code = 0x0000004F, codeExtra = 0x00000000),
		TagInfo(tag = 0x0F000000.toInt(), skey = Hex.decode("bcfe81a3c9d5b9998d0a566c959f3030cc4626795e4eb682ad51391ac42e180ab43161c48a0cc577c6165f322e94d102c48aa30ac60a942a2647036733b12de50721efd2901ec885ba64d1c81dce8dc375a28b940346b80d373647e2dafc74cd663d8e5822e8286d8b541e896df53cf566dbbd0baa86b2c44bbceb2bf41f26fc05e7b8925269eedce542045e217feb8b"), code = 0x00000052, codeExtra = 0x00000000),
		TagInfo(tag = 0x862648D1.toInt(), skey = Hex.decode("98d6bf1124b3f9d7274952dd865b21166dc34a5017b2435847daa0e5e7a173bb35db15293afd5c3705a970bbcaef2b279107962ebb9907eac8e65ab873f7cac941e60e259e4ae7065d894452a555674653af849a744102e11e03baeeceb980ed725f31bc7f062158583031e806e7d0d23e93d8e6b47fd1d7c49650503b0ba5fd3dae35468a9c48eb2d762d4231328b5a"), code = 0x00000052, codeExtra = 0x00000052),
		TagInfo(tag = 0x207BBF2F.toInt(), skey = Hex.decode("0008b533cd5f2ff31f88143c952a8a6ed5effe29e3ea941343d46bbd83c02108d379b3fa65e113e6d354a7f552298b10151e4b0abadeea61df6575550153463bc3ec54ae0933426119ffc970ece50a5b26f19d985f7a989d0e75bc5527ba6ec6e888e92dda0066f7cbdc8203f2f569556212438ed3e38f2887216f659c2ed137b49e532f8e9992a4f75839ed2365e939"), code = 0x0000005A, codeExtra = 0x0000005A),
		TagInfo(tag = 0x09000000.toInt(), skey = Hex.decode("e8531b72c6313efca2a25bf872acf03caba7ee54cbbf59596b83b854131343bccff29e98b236cef0f84cba9831c971e9c85d37a0a02fe50826d40dac01d6e457c7616ec58ab91aeff4f8d9d108a7e95f079df03e8c1a0cfa5cea1ea9c582f4580203802cc3f6e67ebbbb6affd0d01021887a29d3d31200987bc859dc9257dc7fa65d3fdb87b723fcd38e692212e880b6"), code = 0x0000004C, codeExtra = 0x00000000),
		TagInfo(tag = 0xBB67C59F.toInt(), skey = Hex.decode("c757a7943398d39f718350f8290b8b32dab9bc2cc6b91829ba504c94d0e7dcf166390c64083d0bc9ba17adf44bf8a06c677c76f75aa5d3a46a5c084a7170b26bfb388bfab831db3ff296718b4aed9bdb845b6251b481144c08f584f67047b430748eaa93bc79c5908dc86e240212052e2e8474c797d985a1dd3a2b7a6d5b83fe4d188f50134f4cebd393190ed2df96ba"), code = 0x0000005E, codeExtra = 0x0000005E)

	)
}

@Suppress("RemoveRedundantCallsOfConversionMethods", "unused", "MemberVisibilityCanPrivate")
object CryptedPrxKeys16 {
	@Suppress("ArrayInDataClass")
	data class TagInfo2(val tag: Int, val key: ByteArray, val code: Int)

	val keys260_0 = Hex.decode("C32489D38087B24E4CD749E49D1D34D1") // kernel modules 2.60-2.71
	val keys260_1 = Hex.decode("F3AC6E7C040A23E70D33D82473392B4A") // user modules 2.60-2.71
	val keys260_2 = Hex.decode("72B439FF349BAE8230344A1DA2D8B43C") // vshmain 2.60-2.71
	val keys280_0 = Hex.decode("CAFBBFC750EAB4408E445C6353CE80B1") // kernel modules 2.80
	val keys280_1 = Hex.decode("409BC69BA9FB847F7221D23696550974") // user modules 2.80
	val keys280_2 = Hex.decode("03A7CC4A5B91C207FFFC26251E424BB5") // vshmain executable 2.80
	val keys300_0 = Hex.decode("9F671A7A22F3590BAA6DA4C68BD00377") // kernel modules 3.00
	val keys300_1 = Hex.decode("15076326DBE2693456082A934E4B8AB2") // user modules 3.00
	val keys300_2 = Hex.decode("563B69F729882F4CDBD5DE80C65CC873") // vshmain 3.00
	val keys303_0 = Hex.decode("7ba1e25a91b9d31377654ab7c28a10af") // kernel modules 3.00
	val keys310_0 = Hex.decode("a241e839665bfabb1b2d6e0e33e5d73f") // kernel modules 3.10
	val keys310_1 = Hex.decode("A4608FABABDEA5655D433AD15EC3FFEA") // user modules 3.10
	val keys310_2 = Hex.decode("E75C857A59B4E31DD09ECEC2D6D4BD2B") // vshmain 3.10
	val keys310_3 = Hex.decode("2E00F6F752CF955AA126B4849B58762F") // reboot.bin 3.10
	val keys330_0 = Hex.decode("3B9B1A56218014ED8E8B0842FA2CDC3A") // kernel modules 3.30
	val keys330_1 = Hex.decode("E8BE2F06B1052AB9181803E3EB647D26") // user modules 3.30
	val keys330_2 = Hex.decode("AB8225D7436F6CC195C5F7F063733FE7") // vshmain 3.30
	val keys330_3 = Hex.decode("A8B14777DC496A6F384C4D96BD49EC9B") // reboot.bin 3.30
	val keys330_4 = Hex.decode("EC3BD2C0FAC1EEB99ABCFFA389F2601F") // stdio.prx 3.30
	val demokeys_280 = Hex.decode("1299705E24076CD02D06FE7EB30C1126") // demo data.psp 2.80
	val demokeys_3XX_1 = Hex.decode("4705D5E3561E819B092F06DB6B1292E0") // demo data.psp 3.XX
	val demokeys_3XX_2 = Hex.decode("F662396E26224DCA026416997B9AE7B8") // demo data.psp 3.XX
	val ebootbin_271_new = Hex.decode("F4AEF4E186DDD29C7CC542A695A08388") // new 2.7X eboot.bin
	val ebootbin_280_new = Hex.decode("B88C458BB6E76EB85159A6537C5E8631") // new 2.8X eboot.bin
	val ebootbin_300_new = Hex.decode("ED10E036C4FE83F375705EF6A44005F7") // new 3.XX eboot.bin
	val ebootbin_310_new = Hex.decode("5C770CBBB4C24FA27E3B4EB4B4C870AF") // new 3.XX eboot.bin
	val gameshare_260_271 = Hex.decode("F948380C9688A7744F65A054C276D9B8") // 2.60-2.71 gameshare
	val gameshare_280 = Hex.decode("2D86773A56A44FDD3C167193AA8E1143") // 2.80 gameshare
	val gameshare_300 = Hex.decode("781AD28724BDA296183F893672909285") // 3.00 gameshare
	val gameshare_310 = Hex.decode("C97D3E0A54816EC7137499746218E7DD") // 3.10 gameshare
	val keys360_0 = Hex.decode("3C2B51D42D8547DA2DCA18DFFE5409ED") // 3.60 common kernel modules
	val keys360_1 = Hex.decode("311F98D57B58954532AB3AE389324B34") // 3.60 specific slim kernel modules
	val keys370_0 = Hex.decode("26380AACA5D874D132B72ABF799E6DDB") // 3.70 common and fat kernel modules
	val keys370_1 = Hex.decode("53E7ABB9C64A4B779217B5740ADAA9EA") // 3.70 slim specific kernel modules
	val keys370_2 = Hex.decode("7110F0A41614D59312FF7496DF1FDA89") // some 3.70 slim user modules
	val oneseg_310 = Hex.decode("C7277285ABA7F7F04CC186CCE37F17CA") // 1SEG.PBP keys
	val oneseg_300 = Hex.decode("76409E08DB9B3BA1478A968EF3F76292")
	val oneseg_280 = Hex.decode("23DC3BB5A982D6EA63A36E2B2BE9E154")
	val oneseg_260_271 = Hex.decode("224357682F41CE654CA37CC6C4ACF360")
	val oneseg_slim = Hex.decode("12570D8A166D8706037DC88B62A332A9")
	val ms_app_main = Hex.decode("1E2E3849DAD41608272EF3BC37758093")
	val keys390_0 = Hex.decode("45EF5C5DED81998412948FABE8056D7D") // 3.90 kernel
	val keys390_1 = Hex.decode("701B082522A14D3B6921F9710AA841A9") // 3.90 slim
	val keys500_0 = Hex.decode("EB1B530B624932581F830AF4993D75D0") // 5.00 kernel
	val keys500_1 = Hex.decode("BAE2A31207FF041B64A51185F72F995B") // 5.00 kernel 2000 specific
	val keys500_2 = Hex.decode("2C8EAF1DFF79731AAD96AB09EA35598B") // 5.00 kernel 3000 specific
	val keys500_c = Hex.decode("A35D51E656C801CAE377BFCDFF24DA4D")
	val keys505_a = Hex.decode("7B9472274CCC543BAEDF4637AC014D87") // 5.05 kernel specific
	val keys505_0 = Hex.decode("2E8E97A28542707318DAA08AF862A2B0")
	val keys505_1 = Hex.decode("582A4C69197B833DD26161FE14EEAA11")
	val keys02G_E = Hex.decode("9D09FD20F38F10690DB26F00CCC5512E") // for psp 2000 file table and ipl pre-decryption
	val keys03G_E = Hex.decode("4F445C62B353C430FC3AA45BECFE51EA") // for psp 3000 file table and ipl pre-decryption
	val key_D91609F0 = Hex.decode("D036127580562043C430943E1C75D1BF")
	val key_D9160AF0 = Hex.decode("10A9AC16AE19C07E3B607786016FF263")
	val key_D9160BF0 = Hex.decode("8383F13753D0BEFC8DA73252460AC2C2")
	val key_D91611F0 = Hex.decode("61B0C0587157D9FA74670E5C7E6E95B9")
	val key_D91612F0 = Hex.decode("9E20E1CDD788DEC0319B10AFC5B87323") // UMD EBOOT.BIN (OPNSSMP.BIN)
	val key_D91613F0 = Hex.decode("EBFF40D8B41AE166913B8F64B6FCB712")
	val key_2E5E10F0 = Hex.decode("9D5C5BAF8CD8697E519F7096E6D5C4E8") // UMD EBOOT.BIN 2 (OPNSSMP.BIN)
	val key_2E5E12F0 = Hex.decode("8A7BC9D6525888EA518360CA1679E207") // UMD EBOOT.BIN 3 (OPNSSMP.BIN)
	val key_2E5E13F0 = Hex.decode("FFA468C331CAB74CF123FF01653D2636")
	val keys600_u1_457B0BF0 = Hex.decode("7B9472274CCC543BAEDF4637AC014D87")
	val keys600_u1_457B0CF0 = Hex.decode("AC34BAB1978DAE6FBAE8B1D6DFDFF1A2")
	val keys05G_E = Hex.decode("5DAA72F226604D1CE72DC8A32F79C554") // for psp go file table and ipl pre-decryption
	val keys570_5k = Hex.decode("6D72A4BA7FBFD1F1A9F3BB071BC0B366") // 5.70 PSPgo kernel
	val keys620_0 = Hex.decode("D6BDCE1E12AF9AE66930DEDA88B8FFFB") // 6.00-6.20 kernel and phat
	val keys620_1 = Hex.decode("1D13E95004733DD2E1DAB9C1E67B25A7") // 6.00-6.20 slim kernel
	val keys620_3 = Hex.decode("A35D51E656C801CAE377BFCDFF24DA4D")
	val keys620_e = Hex.decode("B1B37F76C3FB88E6F860D3353CA34EF3")
	val keys620_5 = Hex.decode("F1BC1707AEB7C830D8349D406A8EDF4E") // PSPgo internal
	val keys620_5k = Hex.decode("418A354F693ADF04FD3946A25C2DF221") // 6.XX PSPgo kernel
	val keys620_5v = Hex.decode("F28F75A73191CE9E75BD2726B4B40C32")

	val g_tagInfo2 = listOf(
		TagInfo2(tag = 0x380228F0.toInt(), key = keys620_5v, code = 0x5A), // -- PSPgo PSPgo 6.XX vshmain
		TagInfo2(tag = 0x4C942AF0.toInt(), key = keys620_5k, code = 0x43), // PSPgo 6.XX
		TagInfo2(tag = 0x4C9428F0.toInt(), key = keys620_5, code = 0x43), // PSPgo
		TagInfo2(tag = 0x4C9429F0.toInt(), key = keys570_5k, code = 0x43), // PSPgo 5.70
		TagInfo2(tag = 0x4C941DF0.toInt(), key = keys620_1, code = 0x43), // -- 6.00-6.20
		TagInfo2(tag = 0x4C941CF0.toInt(), key = keys620_0, code = 0x43),
		TagInfo2(tag = 0x457B1EF0.toInt(), key = keys620_3, code = 0x5B), // pops_04g.prx
		TagInfo2(tag = 0x457B0BF0.toInt(), key = keys600_u1_457B0BF0, code = 0x5B), // -- 5.55 user modules
		TagInfo2(tag = 0x457B0CF0.toInt(), key = keys600_u1_457B0CF0, code = 0x5B),
		TagInfo2(tag = 0x4C9419F0.toInt(), key = keys500_1, code = 0x43), // -- 5.00 - 5.50
		TagInfo2(tag = 0x4C9418F0.toInt(), key = keys500_0, code = 0x43),
		TagInfo2(tag = 0x4C941FF0.toInt(), key = keys500_2, code = 0x43),
		TagInfo2(tag = 0x4C9417F0.toInt(), key = keys500_1, code = 0x43),
		TagInfo2(tag = 0x4C9416F0.toInt(), key = keys500_0, code = 0x43),
		TagInfo2(tag = 0x4C9414F0.toInt(), key = keys390_0, code = 0x43), // -- 3.90 keys
		TagInfo2(tag = 0x4C9415F0.toInt(), key = keys390_1, code = 0x43),
		TagInfo2(tag = 0xD82310F0.toInt(), key = keys02G_E, code = 0x51), // -- ipl and file tables
		TagInfo2(tag = 0xD8231EF0.toInt(), key = keys03G_E, code = 0x51),
		TagInfo2(tag = 0xD82328F0.toInt(), key = keys05G_E, code = 0x51),
		TagInfo2(tag = 0x4C9412F0.toInt(), key = keys370_0, code = 0x43), // -- 3.60-3.7X keys
		TagInfo2(tag = 0x4C9413F0.toInt(), key = keys370_1, code = 0x43),
		TagInfo2(tag = 0x457B10F0.toInt(), key = keys370_2, code = 0x5B),
		TagInfo2(tag = 0x4C940DF0.toInt(), key = keys360_0, code = 0x43),
		TagInfo2(tag = 0x4C9410F0.toInt(), key = keys360_1, code = 0x43),
		TagInfo2(tag = 0x4C940BF0.toInt(), key = keys330_0, code = 0x43), // -- 3.30-3.51
		TagInfo2(tag = 0x457B0AF0.toInt(), key = keys330_1, code = 0x5B),
		TagInfo2(tag = 0x38020AF0.toInt(), key = keys330_2, code = 0x5A),
		TagInfo2(tag = 0x4C940AF0.toInt(), key = keys330_3, code = 0x43),
		TagInfo2(tag = 0x4C940CF0.toInt(), key = keys330_4, code = 0x43),
		TagInfo2(tag = 0xcfef09f0.toInt(), key = keys310_0, code = 0x62), // -- 3.10
		TagInfo2(tag = 0x457b08f0.toInt(), key = keys310_1, code = 0x5B),
		TagInfo2(tag = 0x380208F0.toInt(), key = keys310_2, code = 0x5A),
		TagInfo2(tag = 0xcfef08f0.toInt(), key = keys310_3, code = 0x62),
		TagInfo2(tag = 0xCFEF07F0.toInt(), key = keys303_0, code = 0x62), // -- 2.60-3.03
		TagInfo2(tag = 0xCFEF06F0.toInt(), key = keys300_0, code = 0x62),
		TagInfo2(tag = 0x457B06F0.toInt(), key = keys300_1, code = 0x5B),
		TagInfo2(tag = 0x380206F0.toInt(), key = keys300_2, code = 0x5A),
		TagInfo2(tag = 0xCFEF05F0.toInt(), key = keys280_0, code = 0x62),
		TagInfo2(tag = 0x457B05F0.toInt(), key = keys280_1, code = 0x5B),
		TagInfo2(tag = 0x380205F0.toInt(), key = keys280_2, code = 0x5A),
		TagInfo2(tag = 0x16D59E03.toInt(), key = keys260_0, code = 0x62),
		TagInfo2(tag = 0x76202403.toInt(), key = keys260_1, code = 0x5B),
		TagInfo2(tag = 0x0F037303.toInt(), key = keys260_2, code = 0x5A),
		TagInfo2(tag = 0x457B28F0.toInt(), key = keys620_e, code = 0x5B),    // -- misc ?
		TagInfo2(tag = 0xADF305F0.toInt(), key = demokeys_280, code = 0x60),    // 2.80 demos data.psp
		TagInfo2(tag = 0xADF306F0.toInt(), key = demokeys_3XX_1, code = 0x60),    // 3.XX demos 1
		TagInfo2(tag = 0xADF308F0.toInt(), key = demokeys_3XX_2, code = 0x60),    // 3.XX demos 2
		TagInfo2(tag = 0x8004FD03.toInt(), key = ebootbin_271_new, code = 0x5D),    // 2.71 eboot.bin
		TagInfo2(tag = 0xD91605F0.toInt(), key = ebootbin_280_new, code = 0x5D),    // 2.80 eboot.bin
		TagInfo2(tag = 0xD91606F0.toInt(), key = ebootbin_300_new, code = 0x5D),    // 3.00 eboot.bin
		TagInfo2(tag = 0xD91608F0.toInt(), key = ebootbin_310_new, code = 0x5D),    // 3.10 eboot.bin
		TagInfo2(tag = 0xD91609F0.toInt(), key = key_D91609F0, code = 0x5D),    // 5.00 eboot.bin
		TagInfo2(tag = 0x2E5E10F0.toInt(), key = key_2E5E10F0, code = 0x48),    // 6.XX eboot.bin
		TagInfo2(tag = 0x2E5E12F0.toInt(), key = key_2E5E12F0, code = 0x48),    // 6.XX eboot.bin
		TagInfo2(tag = 0x2E5E12F0.toInt(), key = key_2E5E13F0, code = 0x48),    // 6.XX eboot.bin
		TagInfo2(tag = 0xD9160AF0.toInt(), key = key_D9160AF0, code = 0x5D),
		TagInfo2(tag = 0xD9160BF0.toInt(), key = key_D9160BF0, code = 0x5D),
		TagInfo2(tag = 0xD91611F0.toInt(), key = key_D91611F0, code = 0x5D),
		TagInfo2(tag = 0xD91612F0.toInt(), key = key_D91612F0, code = 0x5D),
		TagInfo2(tag = 0xD91613F0.toInt(), key = key_D91613F0, code = 0x5D),
		TagInfo2(tag = 0x0A35EA03.toInt(), key = gameshare_260_271, code = 0x5E), // 2.60-2.71 gameshare
		TagInfo2(tag = 0x7B0505F0.toInt(), key = gameshare_280, code = 0x5E),     // 2.80 gameshare
		TagInfo2(tag = 0x7B0506F0.toInt(), key = gameshare_300, code = 0x5E),     // 3.00 gameshare
		TagInfo2(tag = 0x7B0508F0.toInt(), key = gameshare_310, code = 0x5E),     // 3.10+ gameshare
		TagInfo2(tag = 0x279D08F0.toInt(), key = oneseg_310, code = 0x61),     // 3.10 1SEG
		TagInfo2(tag = 0x279D06F0.toInt(), key = oneseg_300, code = 0x61),     // 3.00 1SEG
		TagInfo2(tag = 0x279D05F0.toInt(), key = oneseg_280, code = 0x61),     // 2.80 1SEG
		TagInfo2(tag = 0xD66DF703.toInt(), key = oneseg_260_271, code = 0x61),     // 2.60-2.71 1SEG
		TagInfo2(tag = 0x279D10F0.toInt(), key = oneseg_slim, code = 0x61),     // 02g 1SEG
		TagInfo2(tag = 0x3C2A08F0.toInt(), key = ms_app_main, code = 0x67)     // 1seg ms_application_main.prx
	)
}