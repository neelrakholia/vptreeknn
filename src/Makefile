CC=gcc
CFLAGS= -pg -I.

ssk_dyn: test_ssk_dyn.o ssk_dyn.o 
	gcc $(CFLAGS) -o ssk_dyn test_ssk_dyn.o ssk_dyn.o 

.PHONY: clean

clean:
	rm -f *.o *~
