actual object MyAssert {
    actual fun <T> assertEquals(expect: T?, actual: T?) {
        kotlin.test.assertEquals(expect, actual)
    }
}
