; Assembled into a relocatable object, then given two .text sections away from
; vaddr 0 by llvm-objcopy (see tests/CMakeLists.txt), the higher one first in
; the file.
.text
v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[0:15]   ; 0x2000, 8 bytes
s_barrier                                                 ; 0x2008
s_endpgm                                                  ; 0x200c
.section .text.low,"ax",@progbits                         ; renamed .text
s_nop 0                                                   ; 0x1000
s_mov_b32 s0, 0x12345678                                  ; 0x1004, 8 bytes
s_endpgm                                                  ; 0x100c
