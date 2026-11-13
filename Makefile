# ==============================================================================
# Product:       Logihertz BitNet 1.58-Bit Inference Engine (lh-bitnet-infer)
# Module:        Makefile
# Role:          Production C++17 Build & Test Automation System
# Organization:  Logihertz Systems (OPC) Private Limited
# Developer:     Sameer Tawade (Creator, Founder, Director & Sole Developer)
# Contact:       ops@logihertz.com / security@logihertz.com
# License:       Apache-2.0
# ==============================================================================

CXX ?= g++
ARCH ?= $(shell uname -m)

ifeq ($(ARCH),x86_64)
    SIMD_FLAGS ?= -mavx2
else ifeq ($(ARCH),aarch64)
    SIMD_FLAGS ?= -march=armv8-a+simd
else
    SIMD_FLAGS ?=
endif

CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Werror -pedantic -Iinclude $(SIMD_FLAGS)
LDFLAGS ?=

BIN_DIR = bin
BUILD_DIR = build
SRC_DIR = src
TEST_DIR = tests

TARGET = $(BIN_DIR)/lh-bitnet
TEST_TARGET = $(TEST_DIR)/test_runner

LIB_SRCS = $(SRC_DIR)/bitnet_auditor.cpp \
           $(SRC_DIR)/bitlinear_kernel.cpp \
           $(SRC_DIR)/ternary_quantizer.cpp \
           $(SRC_DIR)/json_parser.cpp \
           $(SRC_DIR)/exporters.cpp

LIB_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(LIB_SRCS))

MAIN_SRC = $(SRC_DIR)/main.cpp
MAIN_OBJ = $(BUILD_DIR)/main.o

TEST_SRC = $(TEST_DIR)/test_runner.cpp
TEST_OBJ = $(BUILD_DIR)/test_runner.o

.PHONY: all test clean dist format check-foreign

all: $(TARGET)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test_runner.o: $(TEST_DIR)/test_runner.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(LIB_OBJS) $(MAIN_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(LIB_OBJS) $(MAIN_OBJ) -o $@ $(LDFLAGS)

$(TEST_TARGET): $(LIB_OBJS) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) $(LIB_OBJS) $(TEST_OBJ) -o $@ $(LDFLAGS)

test: $(TEST_TARGET)
	@echo "Executing Logihertz BitNet C++17 Test Suite..."
	@$(TEST_TARGET)
	@echo "Running CLI Smoke Tests..."
	@$(TARGET) --version
	@$(TARGET) --help > /dev/null
	@$(TARGET) scan fixtures/hardened_bitnet_config.json
	@! $(TARGET) scan fixtures/vulnerable_bitnet_config.json > /dev/null
	@$(TARGET) quantize fixtures/sample_layer_weights.json --output /tmp/test_layer.bnet > /dev/null
	@$(TARGET) infer fixtures/sample_layer_weights.json > /dev/null
	@$(TARGET) benchmark --rows 128 --cols 128 --iterations 10 > /dev/null
	@echo "All tests passed successfully!"

clean:
	@rm -rf $(BUILD_DIR) $(BIN_DIR)/lh-bitnet $(TEST_TARGET) /tmp/test_layer.bnet dist

dist:
	@mkdir -p dist
	tar --exclude='.git' --exclude='build' --exclude='dist' --exclude='bin' --exclude='tests/test_runner' \
	    -czvf dist/lh-bitnet-infer-v0.1.0.tar.gz .
