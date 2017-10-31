# kpspemu
PSP Emulator done in Kotlin Common Platform with Korge targeting JVM and JS for now

[![Build Status](https://travis-ci.org/kpspemu/kpspemu.svg?branch=master)](https://travis-ci.org/kpspemu/kpspemu)

![](/docs/helloworld_js.png)
![](/docs/helloworld_jvm.png)

### Very basic online demo using Kotlin.JS:
* https://kpspemu.github.io/kpspemu-demo/minifire (interpreted)
* https://kpspemu.github.io/kpspemu-demo/helloworld (interpreted)

---

* *REFERENCE:* http://jspspemu.com/#samples/minifire.elf (dynarec) <-- original jspepmu
* *REFERENCE:* http://jspspemu.com/#samples/HelloWorldPSP.elf (dynarec) <-- original jspspemu

### Previous works:
* https://github.com/soywiz/pspemu (PSP Emulator done in D programming language. Interpreted.)
* https://github.com/cspspemu/cspspemu (PSP Emulator done in C# programming language. Dynarec.)
* https://github.com/jspspemu/jspspemu (PSP Emulator done in typescript programming language. Dynarec.)

### Youtube Coding Video Blog

* Vertex Decoder [[Part 1](https://youtu.be/-a6Igq_XiPc)] [[Part 2](https://youtu.be/TZzSfTxDjTo)]
* [Fix Ortho Sample (madd ins + sceCtrl)](https://youtu.be/REF_wFJE85c) 

### Current state:
Right now, this is just a proof of concept. It just runs a very small demo in interpreted mode.

The aim is to create a portable emulator that can run fast in JVM, JS, Android, C++ targets (using libjit).

To achieve this, I have created an embedded module called `dynarek` that will provide an IR that
will generate JS code, JVM bytecode and relevant native code for each supported platform.

The rest of the code is kotlin common and uses [korge](https://github.com/korlibs/korge) and all
the [korlibs](https://github.com/korlibs/korlibs) libraries to do accelerated portable rendering.  

