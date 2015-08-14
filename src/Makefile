CC=gcc
CFLAGS= -g -I.

ssk_dyn: test_ssk_dyn.o ssk_dyn.o 
	gcc -o ssk_dyn test_ssk_dyn.o ssk_dyn.o

.PHONY: clean

clean:
	rm -f *.o *~
