; Assembled into a relocatable object, so .text starts at vaddr 0.
s_nop 0                                                   ; 0x00
v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[0:15]   ; 0x04, 8 bytes
s_barrier                                                 ; 0x0c
s_mov_b32 s0, 0x12345678                                  ; 0x10, 8 bytes
s_endpgm                                                  ; 0x18
