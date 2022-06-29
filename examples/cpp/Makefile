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

.PHONY: all clean
CXX = g++
SRCS = $(wildcard *.cpp)

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

CXX_VERSION:=c++17
GLIB_VERSION:=2.0
GSTREAMER_VERSION:=1.0

DS_SRC_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/sources
DS_INC_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/sources/includes
DS_LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib

INC_INSTALL_DIR?=../../src/

INCS+= $(wildcard ../../src/*.h)

OBJS:= $(SRCS:.c=.o)
OBJS:= $(OBJS:.cpp=.o)
APP:= $(patsubst %.cpp,%.exe,$(SRCS))

CXXFLAGS+= -I$(DS_INC_INSTALL_DIR) \
	-std=$(CXX_VERSION) \
	-I$(INC_INSTALL_DIR) \
	-I$(DS_SRC_INSTALL_DIR)/gst-plugins/gst-nvdspreprocess/include \
	-I/usr/include/gstreamer-$(GSTREAMER_VERSION) \
	-I/usr/include/glib-$(GLIB_VERSION) \
	-I/usr/include/glib-$(GLIB_VERSION)/glib
	
LIBS+= -L$(DS_LIB_INSTALL_DIR) \
	-ldsl \
	-lnvdsgst_meta \
	-lnvds_meta \
	-lglib-$(GLIB_VERSION) \
	-lgstreamer-$(GSTREAMER_VERSION)


PKGS:= gstreamer-$(GSTREAMER_VERSION) \
	glib-$(GLIB_VERSION)

CXXFLAGS+= `pkg-config --cflags $(PKGS)`

LIBS+= `pkg-config --libs $(PKGS)`

all: $(APP)
	
debug: CXXFLAGS += -DDEBUG -g
debug: $(APP)

%.exe : %.cpp $(INCS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm -rf $(APP)
