_ZN9CoolClass3setEi:
##CoolClass :: set(int)
	push	rbp
	mov	rbp, rsp
	mov	QWORD PTR -8[rbp], rdi
	mov	DWORD PTR -12[rbp], esi
	mov	rax, QWORD PTR -8[rbp]
	mov	edx, DWORD PTR -12[rbp]
	mov	DWORD PTR 8[rax], edx
	nop
	pop	rbp
	ret
_ZN9CoolClass3getEv:
## CoolClass :: get()
	push	rbp
	mov	rbp, rsp
	mov	QWORD PTR -8[rbp], rdi
	mov	rax, QWORD PTR -8[rbp]
	mov	eax, DWORD PTR 8[rax]
	pop	rbp
	ret
_ZN13PlainOldClass3setEi:
## PlainOldClass :: set(int)
	push	rbp
	mov	rbp, rsp
	mov	QWORD PTR -8[rbp], rdi
	mov	DWORD PTR -12[rbp], esi
	mov	rax, QWORD PTR -8[rbp]
	mov	edx, DWORD PTR -12[rbp]
	mov	DWORD PTR [rax], edx
	nop
	pop	rbp
	ret
_ZN4BaseC2Ev:
## Base :: Base()
	push	rbp
	mov	rbp, rsp
	mov	QWORD PTR -8[rbp], rdi
	lea	rdx, _ZTV4Base[rip+16]
	mov	rax, QWORD PTR -8[rbp]
	mov	QWORD PTR [rax], rdx
	nop
	pop	rbp
	ret
_ZN9CoolClassC2Ev:
## CoolClass :: CoolClass() 
	push	rbp
	mov	rbp, rsp
	sub	rsp, 16
## novi stack okvir -> 16 B
	mov	QWORD PTR -8[rbp], rdi
	mov	rax, QWORD PTR -8[rbp]
	mov	rdi, rax
## ??? rdi -> -8[rbp] -> rax -> rdi 
## ciklus: zašto??
	call	_ZN4BaseC2Ev
## new Base()
	lea	rdx, _ZTV9CoolClass[rip+16]
	mov	rax, QWORD PTR -8[rbp]
	mov	QWORD PTR [rax], rdx
	nop
	leave
	ret
main:
	push	rbp
	mov	rbp, rsp
	push	rbx
	sub	rsp, 40
* 40 bytes stack frame

	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR -24[rbp], rax
	xor	eax, eax
	mov	edi, 16
	call	_Znwm@PLT   	
* call new
* PLT = Procedure Linkage Table
	  -> used to call external procedures/functions whose address isn't known in the time of linking 

	mov	rbx, rax
	mov	rdi, rbx
	call	_ZN9CoolClassC1Ev
	mov	QWORD PTR -32[rbp], rbx
	lea	rax, -36[rbp]
	mov	esi, 42
## esi -> source index pointer 
	mov	rdi, rax
	call	_ZN13PlainOldClass3setEi
## skok na labelu 
	mov	rax, QWORD PTR -32[rbp]
	mov	rax, QWORD PTR [rax]
	mov	rdx, QWORD PTR -32[rbp]
	mov	esi, 42
	mov	rdi, rdx
	call	rax 
	mov	eax, 0
	mov	rcx, QWORD PTR -24[rbp]
	xor	rcx, QWORD PTR fs:40
	je	.L9
	call	__stack_chk_fail@PLT
.L9:
	add	rsp, 40
	pop	rbx
	pop	rbp
	ret
_ZTV9CoolClass:
## VTable for Cool Class
	.quad	0
	.quad	_ZTI9CoolClass
	.quad	_ZN9CoolClass3setEi
	.quad	_ZN9CoolClass3getEv
	.weak	_ZTV4Base
	.section	.data.rel.ro._ZTV4Base,"awG",@progbits,_ZTV4Base,comdat
	.align 8
	.type	_ZTV4Base, @object
	.size	_ZTV4Base, 32
_ZTV4Base:
## VTable fot Base class
## .quad -> directive is used to define 65 bit numeric value
	.quad	0
	.quad	_ZTI4Base
# Typeinfo for Base
	.quad	__cxa_pure_virtual
	.quad	__cxa_pure_virtual
	.weak	_ZTI9CoolClass
	.section	.data.rel.ro._ZTI9CoolClass,"awG",@progbits,_ZTI9CoolClass,comdat
	.align 8
	.type	_ZTI9CoolClass, @object
	.size	_ZTI9CoolClass, 24
_ZTI9CoolClass:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTS9CoolClass
	.quad	_ZTI4Base
	.weak	_ZTS9CoolClass
	.section	.rodata._ZTS9CoolClass,"aG",@progbits,_ZTS9CoolClass,comdat
	.align 8
	.type	_ZTS9CoolClass, @object
	.size	_ZTS9CoolClass, 11
_ZTS9CoolClass:
	.string	"9CoolClass"
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
	.ident	"GCC: (Ubuntu 7.3.0-27ubuntu1~18.04) 7.3.0"
	.section	.note.GNU-stack,"",@progbits
