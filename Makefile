################################################################################
# 
# The MIT License
# 
# Copyright (c) 2019-Present, ROBERT HOWELL
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

APP:= dslmain

CXX = g++
CC = g++

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=4.0
GS_VERSION:=1.0
GLIB_VERSION:=2.0
GSTREAMER_VERSION:=1.0


SRC_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources
INC_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/includes
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib
CFG_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/samples/configs/deepstream-app
MDL_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/samples/models/Primary_Detector
STR_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/samples/streams

SRCS+= $(wildcard ./src/*.cpp) 

INCS:= $(wildcard ./src/*.h)

PKGS:= gstreamer-$(GSTREAMER_VERSION) \
	gstreamer-video-$(GSTREAMER_VERSION) \
	gstreamer-rtsp-server-$(GSTREAMER_VERSION) \
	x11

OBJS:= $(SRCS:.c=.o)
OBJS:= $(OBJS:.cpp=.o)

ifeq ($(TARGET_DEVICE),aarch64)
	CFLAGS:= -DPLATFORM_TEGRA
endif

CFLAGS+= -I$(INC_INSTALL_DIR) \
    -I$(SRC_INSTALL_DIR)/apps/apps-common/includes \
    -I/opt/include \
	-I/opt/nvidia/deepstream/deepstream-4.0/sources/includes \
	-I/usr/include \
	-I/usr/include/gstreamer-$(GS_VERSION) \
	-I/usr/include/glib-$(GLIB_VERSION) \
	-I/usr/include/glib-$(GLIB_VERSION)/glib \
	-I/usr/lib/aarch64-linux-gnu/glib-$(GLIB_VERSION)/include \
    -DDS_VERSION_MINOR=0 \
    -DDS_VERSION_MAJOR=4 \
	-DDS_CONFIG_DIR='"$(CFG_INSTALL_DIR)"' \
	-DDS_MODELS_DIR='"$(MDL_INSTALL_DIR)"' \
	-DDS_STREAMS_DIR='"$(STR_INSTALL_DIR)"' \
    -DDSL_LOGGER_IMP='"DslLogGst.h"'

LIBS+= -L$(LIB_INSTALL_DIR) \
	-llog4cxx  \
	-laprutil-1 \
	-lapr-1 \
	-lX11 \
	-L/usr/lib/aarch64-linux-gnu \
	-lnvdsgst_meta \
	-lnvds_meta \
	-lnvdsgst_helper \
	-lnvds_utils \
	-lglib-$(GLIB_VERSION) \
	-lgstreamer-$(GSTREAMER_VERSION) \
	-Lgstreamer-video-$(GSTREAMER_VERSION) \
	-Lgstreamer-rtsp-server-$(GSTREAMER_VERSION) \
	-Wl,-rpath,$(LIB_INSTALL_DIR)
	
CFLAGS+= `pkg-config --cflags $(PKGS)`

LIBS+= `pkg-config --libs $(PKGS)`

all: $(APP)

%.o: %.c $(INC_INSTALL_DIR) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	@echo $(SRCS)
	$(CC) -o $(APP) $(OBJS) $(LIBS)

clean:
	rm -rf $(OBJS) $(APP)