import com.soywiz.korio.vfs.localCurrentDirVfs

object KpspTests {
    val root = localCurrentDirVfs["../.."].jail()
    val pspautotests = root["pspautotests"]
    val rootTestResources = root["kpspemu/common/src/test/resources"]
}
