import com.soywiz.korge.gradle.*

buildscript {
    val korgePluginVersion: String by project

    repositories {
        mavenLocal()
        mavenCentral()
        google()
        maven { url = uri("https://plugins.gradle.org/m2/") }
    }
    dependencies {
        classpath("com.soywiz.korlibs.korge.plugins:korge-gradle-plugin:$korgePluginVersion")
    }
}

apply<KorgeGradlePlugin>()

korge {
    id = "com.soywiz.kpspemu"

// To enable all targets at once

    //targetAll()

// To enable targets based on properties/environment variables
    //targetDefault()

// To selectively enable targets

    targetJvm()
    targetJs()
    //targetDesktop()
    //targetIos()
    //targetAndroidIndirect() // targetAndroidDirect()
    //targetAndroidDirect()
}


dependencies {
    //commonMainApi "com.soywiz:korio:$korioVersion"
    //commonMainApi "com.soywiz:korma:$kormaVersion"

    add("jvmMainApi", "org.ow2.asm:asm:6.2.1")
    //add("jvmTestApi", "org.jetbrains.kotlin:kotlin-test-junit:1.6.10")
}

tasks.getByName("jvmTest", Test::class) {
    minHeapSize = "1g"
    maxHeapSize = "2g"
}
