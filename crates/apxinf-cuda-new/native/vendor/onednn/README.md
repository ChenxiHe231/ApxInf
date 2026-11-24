oneDNN 3.12 commit 80afa71049cd69a3df32adcccb623b12cd7baa22.
The staged exp2 tanh GELU follows its SVE injector polynomial and F32
operation boundaries; the table contains rounded exp2(i/64) constants.
Source: https://github.com/uxlfoundation/oneDNN/blob/80afa71049cd69a3df32adcccb623b12cd7baa22/src/cpu/aarch64/injectors/jit_uni_eltwise_injector_sve.cpp
License: Apache 2.0, LICENSE.
