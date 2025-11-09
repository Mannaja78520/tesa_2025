# ====== Compiler and flags ======
CC = gcc
CFLAGS = -Wall -O2

# ====== Find all .c files automatically ======
SRC := $(wildcard *.c)
EXE := $(basename $(SRC))   # เปลี่ยน main.c → main

# ====== Default target ======
# ถ้าไม่ระบุ จะ build ทุกตัว เช่น make → สร้างทุกโปรแกรม
all: $(EXE)

# ====== Rule for building each file ======
%: %.c
	$(CC) $(CFLAGS) -o $@ $<

# ====== Clean ======
clean:
	rm -f $(EXE)
