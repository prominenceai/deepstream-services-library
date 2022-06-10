################################################################################
# 
# The MIT License
# 
# Copyright (c) 2019-2022, Prominence AI, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in-
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
################################################################################

################################################################################
#
# This Makefile can be used to build each of the example cpp files -- one 
# at a time -- in a standalone example app called dsl-example-app.exe
#
# Usage: $ makefile <target> EXAMPLE=<example-cpp-file>
#
# eg:
#      $ makefile debug EXAMPLE=./multiple_pipelines.cpp
#
################################################################################

APP:= dsl-example-app.exe

CXX = g++

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

CXX_VERSION:=c++17
GLIB_VERSION:=2.0

INC_INSTALL_DIR?=../../src/

SRCS+= $(EXAMPLE)
INCS+= $(wildcard ../../src/*.h)

OBJS:= $(SRCS:.c=.o)
OBJS:= $(OBJS:.cpp=.o)

CFLAGS+= -std=$(CXX_VERSION) \
	-I$(INC_INSTALL_DIR) \
	-I/usr/include/glib-$(GLIB_VERSION) \
	-I/usr/include/glib-$(GLIB_VERSION)/glib
	
LIBS+= -ldsl \
	-lglib-$(GLIB_VERSION)

PKGS:= glib-$(GLIB_VERSION)

CFLAGS+= `pkg-config --cflags $(PKGS)`

LIBS+= `pkg-config --libs $(PKGS)`

all: $(APP)
	
debug: CFLAGS += -DDEBUG -g
debug: $(APP)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(APP):	$(OBJS) Makefile
	@echo $(SRCS)
	$(CXX) -o $(APP) $(OBJS) $(LIBS)

clean:
	rm -rf $(OBJS) $(APP)

# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined $1$(if $2, ($2))))