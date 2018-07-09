package com.soywiz.kpspemu.hle.modules

import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.hle.*

@Suppress("UNUSED_PARAMETER")
class pspDveManager(emulator: Emulator) :

    SceModule(emulator, "pspDveManager", 0x00010011, "pspDveManager.prx", "pspDveManager_Library") {

    //STUB_START "pspDveManager",0x40090000,0x00020005
    //STUB_FUNC  0x2ACFCB6D,pspDveMgrCheckVideoOut
    //STUB_FUNC  0xF9C86C73,pspDveMgrSetVideoOut
    //STUB_END

    /*
     *@return 0 - Cable not connected
     *@return 1 - S-Video Cable / AV (composite) cable
     *@return  2 - D Terminal Cable / Component Cable
     *@return < 0 - Error
     */
    fun pspDveMgrCheckVideoOut(): Int {
        return 0 // Cable not connected
    }

    object Cable {
        const val D_TERMINAL_CABLE = 0
        const val S_VIDEO_CABLE = 2
    }

    object Mode {
        const val PROGRESIVE = 0x1D2
        const val INTERLACE = 0x1D1
    }

    //pspDveMgrSetVideoOut(2, 0x1D2, 720, 503, 1, 15, 0); // S-Video Cable / AV (Composite OUT) / Progressive (480p)
    //pspDveMgrSetVideoOut(2, 0x1D1, 720, 503, 1, 15, 0); // S-Video Cable / AV (Composite OUT) / Interlace (480i)
    //pspDveMgrSetVideoOut(0, 0x1D2, 720, 480, 1, 15, 0); // D Terminal Cable (Component OUT) / Progressive (480p)
    //pspDveMgrSetVideoOut(0, 0x1D1, 720, 503, 1, 15, 0); // D Terminal Cable (Component OUT) / Interlace (480i)
    fun pspDveMgrSetVideoOut(cable: Int, mode: Int, width: Int, height: Int, unk2: Int, unk3: Int, unk4: Int): Int {
        return 0
    }


    override fun registerModule() {
        registerFunctionInt("pspDveMgrCheckVideoOut", 0x2ACFCB6D, since = 150) { pspDveMgrCheckVideoOut() }
        registerFunctionInt("pspDveMgrSetVideoOut", 0xF9C86C73, since = 150) {
            pspDveMgrSetVideoOut(
                int,
                int,
                int,
                int,
                int,
                int,
                int
            )
        }
    }
}
