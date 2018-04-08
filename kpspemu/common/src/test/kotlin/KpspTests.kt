import com.soywiz.korio.vfs.*

object KpspTests {
    val root = localCurrentDirVfs["../.."].jail()
    val pspautotests = root["pspautotests"]
    val rootTestResources = root["kpspemu/common/src/test/resources"]
}
