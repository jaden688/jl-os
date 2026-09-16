CC := gcc
CFLAGS := -O3 -Wall -Wextra -std=c99 -g

SOURCES := nhit.c jl_lm.c test.c
OBJECTS := $(SOURCES:.c=.o)
TARGET := test_inference

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^
	@echo "Built: $@"

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

.PHONY: run clean

run: $(TARGET)
	./$(TARGET) mh_model.pt "the "

clean:
	rm -f $(OBJECTS) $(TARGET)

.PRECIOUS: $(OBJECTS)
