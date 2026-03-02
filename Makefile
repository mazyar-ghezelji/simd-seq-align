CXX      := g++
CXXFLAGS := -O3 -std=c++11 -mavx -msse4.2
SRC      := src/main.cpp
BIN      := alignment

.PHONY: all clean run

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)

