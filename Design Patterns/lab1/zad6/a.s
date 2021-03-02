	.file	"a.cpp"
	.intel_syntax noprefix
	.text
	.section	.rodata
.LC0:
	.string	"Base"
	.section	.text._ZN4BaseC2Ev,"axG",@progbits,_ZN4BaseC5Ev,comdat
	.align 2
	.weak	_ZN4BaseC2Ev
	.type	_ZN4BaseC2Ev, @function
_ZN4BaseC2Ev:
.LFB1:
	.cfi_startproc
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	sub	rsp, 16
	mov	QWORD PTR -8[rbp], rdi
	lea	rdx, _ZTV4Base[rip+16]
	mov	rax, QWORD PTR -8[rbp]
	mov	QWORD PTR [rax], rdx
	lea	rdi, .LC0[rip]
	call	puts@PLT
	mov	rax, QWORD PTR -8[rbp]
	mov	rdi, rax
	call	_ZN4Base6metodaEv
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	_ZN4BaseC2Ev, .-_ZN4BaseC2Ev
	.weak	_ZN4BaseC1Ev
	.set	_ZN4BaseC1Ev,_ZN4BaseC2Ev
	.section	.rodata
.LC1:
	.string	"ja sam bazna implementacija!"
	.section	.text._ZN4Base15virtualnaMetodaEv,"axG",@progbits,_ZN4Base15virtualnaMetodaEv,comdat
	.align 2
	.weak	_ZN4Base15virtualnaMetodaEv
	.type	_ZN4Base15virtualnaMetodaEv, @function
_ZN4Base15virtualnaMetodaEv:
.LFB3:
	.cfi_startproc
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	sub	rsp, 16
	mov	QWORD PTR -8[rbp], rdi
	lea	rdi, .LC1[rip]
	call	puts@PLT
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	_ZN4Base15virtualnaMetodaEv, .-_ZN4Base15virtualnaMetodaEv
	.section	.rodata
.LC2:
	.string	"Metoda kaze: "
	.section	.text._ZN4Base6metodaEv,"axG",@progbits,_ZN4Base6metodaEv,comdat
	.align 2
	.weak	_ZN4Base6metodaEv
	.type	_ZN4Base6metodaEv, @function
_ZN4Base6metodaEv:
.LFB4:
	.cfi_startproc
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	sub	rsp, 16
	mov	QWORD PTR -8[rbp], rdi
	lea	rdi, .LC2[rip]
	mov	eax, 0
	call	printf@PLT
	mov	rax, QWORD PTR -8[rbp]
	mov	rax, QWORD PTR [rax]
	mov	rax, QWORD PTR [rax]
	mov	rdx, QWORD PTR -8[rbp]
	mov	rdi, rdx
	call	rax
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	_ZN4Base6metodaEv, .-_ZN4Base6metodaEv
	.section	.rodata
.LC3:
	.string	"Derived"
	.section	.text._ZN7DerivedC2Ev,"axG",@progbits,_ZN7DerivedC5Ev,comdat
	.align 2
	.weak	_ZN7DerivedC2Ev
	.type	_ZN7DerivedC2Ev, @function
_ZN7DerivedC2Ev:
.LFB6:
	.cfi_startproc
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	sub	rsp, 16
	mov	QWORD PTR -8[rbp], rdi
	mov	rax, QWORD PTR -8[rbp]
	mov	rdi, rax
	call	_ZN4BaseC2Ev
	lea	rdx, _ZTV7Derived[rip+16]
## spremi lokaciju vtable u rdx
	mov	rax, QWORD PTR -8[rbp]
	mov	QWORD PTR [rax], rdx
## rdx(p na vtab) ide u memoriju
	lea	rdi, .LC3[rip]
	call	puts@PLT
	mov	rax, QWORD PTR -8[rbp]
	mov	rdi, rax
	call	_ZN4Base6metodaEv
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	_ZN7DerivedC2Ev, .-_ZN7DerivedC2Ev
	.weak	_ZN7DerivedC1Ev
	.set	_ZN7DerivedC1Ev,_ZN7DerivedC2Ev
	.section	.rodata
	.align 8
.LC4:
	.string	"ja sam izvedena implementacija!"
	.section	.text._ZN7Derived15virtualnaMetodaEv,"axG",@progbits,_ZN7Derived15virtualnaMetodaEv,comdat
	.align 2
	.weak	_ZN7Derived15virtualnaMetodaEv
	.type	_ZN7Derived15virtualnaMetodaEv, @function
_ZN7Derived15virtualnaMetodaEv:
.LFB8:
	.cfi_startproc
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	sub	rsp, 16
	mov	QWORD PTR -8[rbp], rdi
	lea	rdi, .LC4[rip]
	call	puts@PLT
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	_ZN7Derived15virtualnaMetodaEv, .-_ZN7Derived15virtualnaMetodaEv
	.text
	.globl	main
	.type	main, @function
main:
.LFB9:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA9
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	r12
	push	rbx
	sub	rsp, 16
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	mov	edi, 8
.LEHB0:
	call	_Znwm@PLT
.LEHE0:
	mov	rbx, rax
	mov	rdi, rbx
.LEHB1:
	call	_ZN7DerivedC1Ev
.LEHE1:
	mov	QWORD PTR -24[rbp], rbx
	mov	rax, QWORD PTR -24[rbp]
	mov	rdi, rax
.LEHB2:
	call	_ZN4Base6metodaEv
	mov	eax, 0
	jmp	.L10
.L9:
	mov	r12, rax
	mov	esi, 8
	mov	rdi, rbx
	call	_ZdlPvm@PLT
	mov	rax, r12
	mov	rdi, rax
	call	_Unwind_Resume@PLT
.LEHE2:
.L10:
	add	rsp, 16
	pop	rbx
	pop	r12
	pop	rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.globl	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
.LLSDA9:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE9-.LLSDACSB9
.LLSDACSB9:
	.uleb128 .LEHB0-.LFB9
	.uleb128 .LEHE0-.LEHB0
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB1-.LFB9
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L9-.LFB9
	.uleb128 0
	.uleb128 .LEHB2-.LFB9
	.uleb128 .LEHE2-.LEHB2
	.uleb128 0
	.uleb128 0
.LLSDACSE9:
	.text
	.size	main, .-main
	.weak	_ZTV7Derived
	.section	.data.rel.ro.local._ZTV7Derived,"awG",@progbits,_ZTV7Derived,comdat
	.align 8
	.type	_ZTV7Derived, @object
	.size	_ZTV7Derived, 24

VIRTUALNE TABLICE
----------------------------------------------
_ZTV7Derived:
	.quad	0
	.quad	_ZTI7Derived
	.quad	_ZN7Derived15virtualnaMetodaEv
	.weak	_ZTV4Base
	.section	.data.rel.ro.local._ZTV4Base,"awG",@progbits,_ZTV4Base,comdat
	.align 8
	.type	_ZTV4Base, @object
	.size	_ZTV4Base, 24
_ZTV4Base:
	.quad	0
	.quad	_ZTI4Base
	.quad	_ZN4Base15virtualnaMetodaEv
	.weak	_ZTI7Derived
	.section	.data.rel.ro._ZTI7Derived,"awG",@progbits,_ZTI7Derived,comdat
	.align 8
	.type	_ZTI7Derived, @object
	.size	_ZTI7Derived, 24
----------------------------------------------


_ZTI7Derived:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTS7Derived
	.quad	_ZTI4Base
	.weak	_ZTS7Derived
	.section	.rodata._ZTS7Derived,"aG",@progbits,_ZTS7Derived,comdat
	.align 8
	.type	_ZTS7Derived, @object
	.size	_ZTS7Derived, 9
_ZTS7Derived:
	.string	"7Derived"
	.weak	_ZTI4Base
	.section	.data.rel.ro._ZTI4Base,"awG",@progbits,_ZTI4Base,comdat
	.align 8
	.type	_ZTI4Base, @object
	.size	_ZTI4Base, 16
_ZTI4Base:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTS4Base
	.weak	_ZTS4Base
	.section	.rodata._ZTS4Base,"aG",@progbits,_ZTS4Base,comdat
	.type	_ZTS4Base, @object
	.size	_ZTS4Base, 6
_ZTS4Base:
	.string	"4Base"
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.ident	"GCC: (Ubuntu 7.3.0-27ubuntu1~18.04) 7.3.0"
	.section	.note.GNU-stack,"",@progbits
