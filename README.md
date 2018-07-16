# kpspemu

PSP Emulator done in Kotlin Common Platform with Korge targeting JVM and JS for now

[![Build Status](https://travis-ci.org/kpspemu/kpspemu.svg?branch=master)](https://travis-ci.org/kpspemu/kpspemu)

![](/docs/kpspemu-logo-200x200.png)

[![](/docs/cube.png)](https://kpspemu.github.io/kpspemu-demo/cube/)
[![](/docs/0.3.2-SNAPSHOT-JVM.png)](https://kpspemu.github.io/kpspemu-demo/0.3.2/)

### Running from source-code:

JVM: `./gradlew runApp`
JS: `./gradlew compileKotlin2Js && http-server kpspemu/js/web`

Or open `build.gradle` with intelliJ and open `kpspemu/common/src/com/soywiz/kpspemu/Main.kt` and execute the fun `main` method

### Running tests:

```
./gradlew check
```

### More advanced homebrew working:
* https://kpspemu.github.io/kpspemu-demo/0.3.1/ (interpreted)
* https://kpspemu.github.io/kpspemu-demo/0.3.1/#samples/cube.cso (interpreted)
* https://kpspemu.github.io/kpspemu-demo/0.3.1/#samples/TrigWars.zip (interpreted)
* https://kpspemu.github.io/kpspemu-demo/0.3.1/#samples/cavestory.zip (interpreted)
* *More coming soon...*

### Previous works:
* https://github.com/soywiz/pspemu (PSP Emulator done in D programming language. Interpreted.)
* https://github.com/cspspemu/cspspemu (PSP Emulator done in C# programming language. Dynarec.)
* https://github.com/jspspemu/jspspemu (PSP Emulator done in typescript programming language. Dynarec.)

### Youtube Coding Video Blog

* Vertex Decoder [[Part 1](https://youtu.be/-a6Igq_XiPc)] [[Part 2](https://youtu.be/TZzSfTxDjTo)]
* [Fix Ortho Sample (madd ins + sceCtrl)](https://youtu.be/REF_wFJE85c)
* 2017-12-12: [Splitting ThreadManForUser in intelliJ](https://www.youtube.com/watch?v=fdcpPWjxl1A)

### Current state:
Right now it is capable to run some homebrew in interpreted mode and starts to run some early simple commercial games.

The aim is to create a portable emulator that can run fast in
JVM (generating bytecode),
JS (generating JavaScript),
Android (generating dex or in interpreted mode),
C++ targets (using libjit or in interpreted mode).

To achieve this, I have created a library called [`dynarek`](https://korlibs.github.io/dynarek/) that will provide an IR that
will generate JS code, JVM bytecode and relevant native code for each supported platform.

The rest of the code is kotlin common and uses [korge](https://github.com/korlibs/korge) and all
the [korlibs](https://github.com/korlibs/) libraries to do accelerated portable rendering, input,
audio, ui, timers, logging, zlib...  
