all: main

CXXFLAGS := -g -std=c++20 -Wall -Wextra
LDLIBS := -lSDL2 -lfmt

clean:
	rm main main_old
