package com.soywiz.kpspemu.hle.modules


import com.soywiz.kpspemu.*
import com.soywiz.kpspemu.cpu.*
import com.soywiz.kpspemu.hle.*


class sceNetInet(emulator: Emulator) :
    SceModule(emulator, "sceNetInet", 0x00010011, "pspnet_inet.prx", "sceNetInet_Library") {
    fun sceNetInetSendto(cpu: CpuState): Unit = UNIMPLEMENTED(0x05038FC7)
    fun sceNetInetGetsockname(cpu: CpuState): Unit = UNIMPLEMENTED(0x162E6FD5)
    fun sceNetInetInit(cpu: CpuState): Unit = UNIMPLEMENTED(0x17943399)
    fun sceNetInetBind(cpu: CpuState): Unit = UNIMPLEMENTED(0x1A33F9AE)
    fun sceNetInetInetAton(cpu: CpuState): Unit = UNIMPLEMENTED(0x1BDF5D13)
    fun sceNetInet_1D023504(cpu: CpuState): Unit = UNIMPLEMENTED(0x1D023504)
    fun sceNetInet_2D5868C0(cpu: CpuState): Unit = UNIMPLEMENTED(0x2D5868C0)
    fun sceNetInetSetsockopt(cpu: CpuState): Unit = UNIMPLEMENTED(0x2FE71FE7)
    fun sceNetInetGetUdpcbstat(cpu: CpuState): Unit = UNIMPLEMENTED(0x39B0C7D3)
    fun sceNetInetConnect(cpu: CpuState): Unit = UNIMPLEMENTED(0x410B34AA)
    fun sceNetInetGetsockopt(cpu: CpuState): Unit = UNIMPLEMENTED(0x4A114C7C)
    fun sceNetInetShutdown(cpu: CpuState): Unit = UNIMPLEMENTED(0x4CFE4E56)
    fun sceNetInetSelect(cpu: CpuState): Unit = UNIMPLEMENTED(0x5BE8D595)
    fun sceNetInetSendmsg(cpu: CpuState): Unit = UNIMPLEMENTED(0x774E36F4)
    fun sceNetInetSend(cpu: CpuState): Unit = UNIMPLEMENTED(0x7AA671BC)
    fun sceNetInetCloseWithRST(cpu: CpuState): Unit = UNIMPLEMENTED(0x805502DD)
    fun sceNetInetSocketAbort(cpu: CpuState): Unit = UNIMPLEMENTED(0x80A21ABD)
    fun sceNetInetSocket(cpu: CpuState): Unit = UNIMPLEMENTED(0x8B7B220F)
    fun sceNetInetGetPspError(cpu: CpuState): Unit = UNIMPLEMENTED(0x8CA3A97E)
    fun sceNetInetClose(cpu: CpuState): Unit = UNIMPLEMENTED(0x8D7284EA)
    fun sceNetInetTerm(cpu: CpuState): Unit = UNIMPLEMENTED(0xA9ED66B9)
    fun sceNetInetGetTcpcbstat(cpu: CpuState): Unit = UNIMPLEMENTED(0xB3888AD4)
    fun sceNetInetInetAddr(cpu: CpuState): Unit = UNIMPLEMENTED(0xB75D5B0A)
    fun sceNetInetRecvfrom(cpu: CpuState): Unit = UNIMPLEMENTED(0xC91142E4)
    fun sceNetInet_CCC18C45(cpu: CpuState): Unit = UNIMPLEMENTED(0xCCC18C45)
    fun sceNetInetRecv(cpu: CpuState): Unit = UNIMPLEMENTED(0xCDA85C99)
    fun sceNetInetInetNtop(cpu: CpuState): Unit = UNIMPLEMENTED(0xD0792666)
    fun sceNetInetListen(cpu: CpuState): Unit = UNIMPLEMENTED(0xD10A1A7A)
    fun sceNetInetAccept(cpu: CpuState): Unit = UNIMPLEMENTED(0xDB094E1B)
    fun sceNetInetGetpeername(cpu: CpuState): Unit = UNIMPLEMENTED(0xE247B6D6)
    fun sceNetInetInetPton(cpu: CpuState): Unit = UNIMPLEMENTED(0xE30B8C19)
    fun sceNetInetRecvmsg(cpu: CpuState): Unit = UNIMPLEMENTED(0xEECE61D2)
    fun sceNetInetPoll(cpu: CpuState): Unit = UNIMPLEMENTED(0xFAABB1DD)
    fun sceNetInetGetErrno(cpu: CpuState): Unit = UNIMPLEMENTED(0xFBABE411)


    override fun registerModule() {
        registerFunctionRaw("sceNetInetSendto", 0x05038FC7, since = 150) { sceNetInetSendto(it) }
        registerFunctionRaw("sceNetInetGetsockname", 0x162E6FD5, since = 150) { sceNetInetGetsockname(it) }
        registerFunctionRaw("sceNetInetInit", 0x17943399, since = 150) { sceNetInetInit(it) }
        registerFunctionRaw("sceNetInetBind", 0x1A33F9AE, since = 150) { sceNetInetBind(it) }
        registerFunctionRaw("sceNetInetInetAton", 0x1BDF5D13, since = 150) { sceNetInetInetAton(it) }
        registerFunctionRaw("sceNetInet_1D023504", 0x1D023504, since = 150) { sceNetInet_1D023504(it) }
        registerFunctionRaw("sceNetInet_2D5868C0", 0x2D5868C0, since = 150) { sceNetInet_2D5868C0(it) }
        registerFunctionRaw("sceNetInetSetsockopt", 0x2FE71FE7, since = 150) { sceNetInetSetsockopt(it) }
        registerFunctionRaw("sceNetInetGetUdpcbstat", 0x39B0C7D3, since = 150) { sceNetInetGetUdpcbstat(it) }
        registerFunctionRaw("sceNetInetConnect", 0x410B34AA, since = 150) { sceNetInetConnect(it) }
        registerFunctionRaw("sceNetInetGetsockopt", 0x4A114C7C, since = 150) { sceNetInetGetsockopt(it) }
        registerFunctionRaw("sceNetInetShutdown", 0x4CFE4E56, since = 150) { sceNetInetShutdown(it) }
        registerFunctionRaw("sceNetInetSelect", 0x5BE8D595, since = 150) { sceNetInetSelect(it) }
        registerFunctionRaw("sceNetInetSendmsg", 0x774E36F4, since = 150) { sceNetInetSendmsg(it) }
        registerFunctionRaw("sceNetInetSend", 0x7AA671BC, since = 150) { sceNetInetSend(it) }
        registerFunctionRaw("sceNetInetCloseWithRST", 0x805502DD, since = 150) { sceNetInetCloseWithRST(it) }
        registerFunctionRaw("sceNetInetSocketAbort", 0x80A21ABD, since = 150) { sceNetInetSocketAbort(it) }
        registerFunctionRaw("sceNetInetSocket", 0x8B7B220F, since = 150) { sceNetInetSocket(it) }
        registerFunctionRaw("sceNetInetGetPspError", 0x8CA3A97E, since = 150) { sceNetInetGetPspError(it) }
        registerFunctionRaw("sceNetInetClose", 0x8D7284EA, since = 150) { sceNetInetClose(it) }
        registerFunctionRaw("sceNetInetTerm", 0xA9ED66B9, since = 150) { sceNetInetTerm(it) }
        registerFunctionRaw("sceNetInetGetTcpcbstat", 0xB3888AD4, since = 150) { sceNetInetGetTcpcbstat(it) }
        registerFunctionRaw("sceNetInetInetAddr", 0xB75D5B0A, since = 150) { sceNetInetInetAddr(it) }
        registerFunctionRaw("sceNetInetRecvfrom", 0xC91142E4, since = 150) { sceNetInetRecvfrom(it) }
        registerFunctionRaw("sceNetInet_CCC18C45", 0xCCC18C45, since = 150) { sceNetInet_CCC18C45(it) }
        registerFunctionRaw("sceNetInetRecv", 0xCDA85C99, since = 150) { sceNetInetRecv(it) }
        registerFunctionRaw("sceNetInetInetNtop", 0xD0792666, since = 150) { sceNetInetInetNtop(it) }
        registerFunctionRaw("sceNetInetListen", 0xD10A1A7A, since = 150) { sceNetInetListen(it) }
        registerFunctionRaw("sceNetInetAccept", 0xDB094E1B, since = 150) { sceNetInetAccept(it) }
        registerFunctionRaw("sceNetInetGetpeername", 0xE247B6D6, since = 150) { sceNetInetGetpeername(it) }
        registerFunctionRaw("sceNetInetInetPton", 0xE30B8C19, since = 150) { sceNetInetInetPton(it) }
        registerFunctionRaw("sceNetInetRecvmsg", 0xEECE61D2, since = 150) { sceNetInetRecvmsg(it) }
        registerFunctionRaw("sceNetInetPoll", 0xFAABB1DD, since = 150) { sceNetInetPoll(it) }
        registerFunctionRaw("sceNetInetGetErrno", 0xFBABE411, since = 150) { sceNetInetGetErrno(it) }
    }
}
