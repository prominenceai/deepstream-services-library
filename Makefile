################################################################################
# 
# The MIT License
# 
# Copyright (c) 2019-2021, Prominence AI, Inc.
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


APP:= dsl-test-app

CXX = g++

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

DSL_VERSION:='L"v0.09.alpha"'
NVDS_VERSION:=5.0
GS_VERSION:=1.0
GLIB_VERSION:=2.0
GSTREAMER_VERSION:=1.0
CUDA_VERSION:=10.2

SRC_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources
INC_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/includes
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

SRCS+= $(wildcard ./src/*.cpp)
SRCS+= $(wildcard ./test/*.cpp)
SRCS+= $(wildcard ./test/api/*.cpp)
SRCS+= $(wildcard ./test/unit/*.cpp)

INCS:= $(wildcard ./src/*.h)
INCS+= $(wildcard ./test/*.hpp)


TEST_OBJS+= $(wildcard ./test/api/*.o)
TEST_OBJS+= $(wildcard ./test/unit/*.o)

PKGS:= gstreamer-$(GSTREAMER_VERSION) \
	gstreamer-video-$(GSTREAMER_VERSION) \
	gstreamer-rtsp-server-$(GSTREAMER_VERSION) \
	x11 \
	opencv4

OBJS:= $(SRCS:.c=.o)
OBJS:= $(OBJS:.cpp=.o)

ifeq ($(TARGET_DEVICE),aarch64)
	CFLAGS:= -DPLATFORM_TEGRA
endif

CFLAGS+= -I$(INC_INSTALL_DIR) \
    -I$(SRC_INSTALL_DIR)/apps/apps-common/includes \
    -I/opt/include \
	-I/usr/include \
	-I/usr/include/gstreamer-$(GS_VERSION) \
	-I/usr/include/glib-$(GLIB_VERSION) \
	-I/usr/include/glib-$(GLIB_VERSION)/glib \
	-I/usr/lib/aarch64-linux-gnu/glib-$(GLIB_VERSION)/include \
	-I/usr/local/cuda-$(CUDA_VERSION)/targets/aarch64-linux/include \
	-I./src \
	-I./test \
	-I./test/api \
	-DDSL_VERSION=$(DSL_VERSION) \
    -DDS_VERSION_MINOR=0 \
    -DDS_VERSION_MAJOR=4 \
    -DDSL_LOGGER_IMP='"DslLogGst.h"'\
	-DNVDS_KLT_LIB='"$(LIB_INSTALL_DIR)/libnvds_mot_klt.so"' \
	-DNVDS_IOU_LIB='"$(LIB_INSTALL_DIR)/libnvds_mot_iou.so"' \
    -fPIC 

LIBS+= -L$(LIB_INSTALL_DIR) \
	-laprutil-1 \
	-lapr-1 \
	-lX11 \
	-L/usr/lib/aarch64-linux-gnu \
	-lnvdsgst_meta \
	-lnvds_meta \
	-lnvdsgst_helper \
	-lnvds_utils \
	-lnvbufsurface \
	-lnvbufsurftransform \
	-lnvdsgst_smartrecord \
	-lglib-$(GLIB_VERSION) \
	-lgstreamer-$(GSTREAMER_VERSION) \
	-Lgstreamer-video-$(GSTREAMER_VERSION) \
	-Lgstreamer-rtsp-server-$(GSTREAMER_VERSION) \
	-L/usr/local/cuda-$(CUDA_VERSION)/lib64/ -lcudart \
	-Wl,-rpath,$(LIB_INSTALL_DIR)
	
CFLAGS+= `pkg-config --cflags $(PKGS)`

LIBS+= `pkg-config --libs $(PKGS)`

all: $(APP)

debug: CFLAGS += -DDEBUG -g
debug: $(APP)

PCH_INC=./src/Dsl.h
PCH_OUT=./src/Dsl.h.gch
$(PCH_OUT): $(PCH_INC) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cpp $(PCH_OUT) $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	@echo $(SRCS)
	$(CXX) -o $(APP) $(OBJS) $(LIBS)

lib:
	ar rcs dsl-lib.a $(OBJS)
	ar dv dsl-lib.a DslCatch.o $(TEST_OBJS)
	$(CXX) -shared $(OBJS) -o dsl-lib.so $(LIBS)
	cp dsl-lib.so examples/python/
	
so_lib:
	$(CXX) -shared $(OBJS) -o dsl-lib.so $(LIBS) 
	cp dsl-lib.so examples/python/

clean:
	rm -rf $(OBJS) $(APP) dsl-lib.a dsl-lib.so $(PCH_OUT)
