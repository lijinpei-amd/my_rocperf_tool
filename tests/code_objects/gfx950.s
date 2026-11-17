; Assembled into a relocatable object, so .text starts at vaddr 0. Each MFMA's
; comment gives its passes in LLVM's gfx950 scheduling model.
v_mfma_f32_32x32x16_bf16 a[0:15], v[0:3], v[4:7], a[0:15]     ; 0x00, 8 passes
v_mfma_f32_32x32x16_f16 a[0:15], v[0:3], v[4:7], a[0:15]      ; 0x08, 8 passes
v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[0:15]       ; 0x10, 8 passes
v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], v[4:7], a[0:3]       ; 0x18, 4 passes
v_mfma_f64_16x16x4_f64 v[0:7], v[8:9], v[10:11], v[0:7]       ; 0x20, 16 passes
; A variant scheduling class, resolved by the instruction's operands.
v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3]   ; 0x28, 4 passes
v_mfma_ld_scale_b32 v0, v1                                    ; 0x30, not an MFMA
s_endpgm                                                      ; 0x38
