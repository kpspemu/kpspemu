import org.junit.*

actual object MyAssert {
    actual fun <T> assertEquals(expect: T?, actual: T?) {
        Assert.assertEquals(expect, actual)
    }
}
