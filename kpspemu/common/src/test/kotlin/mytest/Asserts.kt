package mytest

fun <T> assertEquals(expected: T, actual: T) {
    MyAssert.assertEquals(expected, actual)
}