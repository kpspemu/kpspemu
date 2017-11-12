package com.soywiz.dynarek

expect fun <TRet> DFunction0<TRet>.generateDynarek(): () -> TRet
expect fun <TRet, T0> DFunction1<TRet, T0>.generateDynarek(): (T0) -> TRet
expect fun <TRet, T0, T1> DFunction2<TRet, T0, T1>.generateDynarek(): (T0, T1) -> TRet

