################################################################################
# 
# The MIT License
# 
# Copyright (c) 2019-2024, Prominence AI, Inc.
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


APP:= dsl-test-app.exe
LIB:= libdsl

CXX = g++

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

ifeq ($(SUDO_USER),)
	USER_SITE = "`python3 -m site --user-site`"
else
	USER_SITE = "`sudo -u ${SUDO_USER} python3 -m site --user-site`"
endif

CXX_VERSION:=c++17
DSL_VERSION:='L"v0.31.c.alpha"'

GLIB_VERSION:=2.0
GSTREAMER_VERSION:=1.0
GSTREAMER_SUB_VERSION:=20
GSTREAMER_SDP_VERSION:=1.0
GSTREAMER_WEBRTC_VERSION:=1.0
LIBSOUP_VERSION:=2.4
JSON_GLIB_VERSION:=1.0

# To enable the extended Image Services, ensure FFmpeg or OpenCV 
# is installed (See /docs/installing-dependencies.md), and
#  - set either BUILD_WITH_FFMPEG or BUILD_WITH_OPENCV:=true (NOT both)
BUILD_WITH_FFMPEG:=false
BUILD_WITH_OPENCV:=false

# To enable the InterPipe Sink and Source components
# - set BUILD_INTER_PIPE:=true
BUILD_INTER_PIPE:=false

# To enable the WebRTC Sink component (requires GStreamer >= 1.20)
# - set BUILD_WEBRTC:=true
BUILD_WEBRTC:=false

# To enable the LiveKit WebRTC Sink component (requires GStreamer >= 1.22)
# - set BUILD_LIVEKIT_WEBRTC:=true
BUILD_LIVEKIT_WEBRTC:=false

# To enable the Non Maximum Processor (NMP) Pad Probe Handler (PPH)
# - set BUILD_NMP_PPH:=true and NUM_CPP_PATH:=<path-to-numcpp-include-folder>
BUILD_NMP_PPH:=false
NUM_CPP_PATH:=

# Fail if both build flags are set
ifeq ($(BUILD_WITH_FFMPEG),true)
ifeq ($(BUILD_WITH_OPENCV),true)
$(error BUILD_WITH_FFMPEG and BUILD_WITH_OPENCV both set to true)
endif
endif

SRC_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/sources
INC_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/sources/includes
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib

ifeq ($(BUILD_WITH_FFMPEG),true)
SRCS+= $(wildcard ./src/ffmpeg/*.cpp)
SRCS+= $(wildcard ./test/avfile/*.cpp)
INCS+= $(wildcard ./src/ffmpeg/*.h)
endif

ifeq ($(BUILD_WITH_OPENCV),true)
SRCS+= $(wildcard ./src/opencv/*.cpp)
SRCS+= $(wildcard ./test/avfile/*.cpp)
INCS+= $(wildcard ./src/opencv/*.h)
endif

ifeq ($(BUILD_INTER_PIPE),true)
SRCS+= $(wildcard ./test/interpipe/*.cpp)
endif

ifeq ($(BUILD_LIVEKIT_WEBRTC),true)
SRCS+= $(wildcard ./test/livekitwebrtc/*.cpp)
endif

ifeq ($(BUILD_NMP_PPH),true)
SRCS+= $(wildcard ./src/nmp/*.cpp)
SRCS+= $(wildcard ./test/nmp/*.cpp)
INCS+= $(wildcard ./src/nmp/*.h)
endif

SRCS+= $(wildcard ./src/*.cpp)
SRCS+= $(wildcard ./src/thirdparty/*.cpp)
SRCS+= $(wildcard ./test/*.cpp)
SRCS+= $(wildcard ./test/api/*.cpp)
SRCS+= $(wildcard ./test/behavior/*.cpp)
SRCS+= $(wildcard ./test/unit/*.cpp)

INCS+= $(wildcard ./src/*.h)
INCS+= $(wildcard ./src/thirdparty/*.h)
INCS+= $(wildcard ./test/*.hpp)

TEST_OBJS+= $(wildcard ./test/api/*.o)
TEST_OBJS+= $(wildcard ./test/behavior/*.o)
TEST_OBJS+= $(wildcard ./test/unit/*.o)

ifeq ($(BUILD_WEBRTC),true)
SRCS+= $(wildcard ./src/webrtc/*.cpp)
SRCS+= $(wildcard ./test/webrtc/*.cpp)
INCS+= $(wildcard ./src/webrtc/*.h)
TEST_OBJS+= $(wildcard ./test/webrtc/*.o)
endif

ifeq ($(BUILD_NMP_PPH),true)
TEST_OBJS+= $(wildcard ./test/nmp/*.o)
endif


OBJS:= $(SRCS:.c=.o)
OBJS:= $(OBJS:.cpp=.o)

CFLAGS+= -I$(INC_INSTALL_DIR) \
	-std=$(CXX_VERSION) \
	-Wno-deprecated-declarations \
	-I$(SRC_INSTALL_DIR)/apps/apps-common/includes \
	-I/opt/include \
	-I/usr/include \
	-I/usr/include/gstreamer-$(GSTREAMER_VERSION) \
	-I/usr/include/glib-$(GLIB_VERSION) \
	-I/usr/include/glib-$(GLIB_VERSION)/glib \
	-I/usr/lib/$(TARGET_DEVICE)-linux-gnu/glib-$(GLIB_VERSION)/include \
	-I/usr/local/cuda/targets/$(TARGET_DEVICE)-linux/include \
	-I./src \
	-I./src/thirdparty \
	-I./test \
	-I./test/api \
	-DDSL_VERSION=$(DSL_VERSION) \
	-DDSL_LOGGER_IMP='"DslLogGst.h"'\
	-DBUILD_WITH_FFMPEG=$(BUILD_WITH_FFMPEG) \
	-DBUILD_WITH_OPENCV=$(BUILD_WITH_OPENCV) \
	-DBUILD_INTER_PIPE=$(BUILD_INTER_PIPE) \
	-DBUILD_WEBRTC=$(BUILD_WEBRTC) \
	-DBUILD_LIVEKIT_WEBRTC=$(BUILD_LIVEKIT_WEBRTC) \
	-DBUILD_NMP_PPH=$(BUILD_NMP_PPH) \
	-DBUILD_MESSAGE_SINK=$(BUILD_MESSAGE_SINK) \
	-DNVDS_MOT_LIB='"$(LIB_INSTALL_DIR)/libnvds_nvmultiobjecttracker.so"' \
	-DNVDS_AMQP_PROTO_LIB='L"$(LIB_INSTALL_DIR)/libnvds_amqp_proto.so"' \
	-DNVDS_AZURE_PROTO_LIB='L"$(LIB_INSTALL_DIR)/libnvds_azure_proto.so"' \
	-DNVDS_AZURE_EDGE_PROTO_LIB='L"$(LIB_INSTALL_DIR)/libnvds_azure_edge_proto"' \
	-DNVDS_KAFKA_PROTO_LIB='L"$(LIB_INSTALL_DIR)/libnvds_kafka_proto.so"' \
	-DNVDS_REDIS_PROTO_LIB='L"$(LIB_INSTALL_DIR)/libnvds_redis_proto.so"' \
    -fPIC 

ifeq ($(BUILD_WITH_FFMPEG),true)
CFLAGS+= -I./src/ffmpeg \
	-I./test/avfile
endif	

ifeq ($(BUILD_WITH_OPENCV),true)
CFLAGS+= -I /usr/include/opencv4 \
	-I./src/opencv/ \
	-I./test/avfile
endif	

ifeq ($(BUILD_LIVEKIT_WEBRTC),true)
CFLAGS+= -I./test/livekitwebrtc
endif	

ifeq ($(BUILD_WEBRTC),true)
CFLAGS+= -I/usr/include/libsoup-$(LIBSOUP_VERSION) \
	-I/usr/include/json-glib-$(JSON_GLIB_VERSION) \
	-I./src/webrtc
endif	

ifeq ($(BUILD_NMP_PPH),true)
CFLAGS+= -I./src/nmp \
	-I$(NUM_CPP_PATH) \
	-DNUMCPP_NO_USE_BOOST
endif	

CFLAGS += `geos-config --cflags`	

LIBS+= -L$(LIB_INSTALL_DIR) \
	-L/usr/local/lib \
	-laprutil-1 \
	-lapr-1 \
	-lX11 \
	-lcuda \
	-L/usr/lib/$(TARGET_DEVICE)-linux-gnu \
	-lgeos_c \
	-lcurl \
	-lnvdsgst_meta \
	-lnvds_meta \
	-lnvdsgst_helper \
	-lnvds_utils \
	-lnvbufsurface \
	-lnvbufsurftransform \
	-lnvdsgst_smartrecord \
	-lnvds_msgbroker \
	-lglib-$(GLIB_VERSION) \
	-lgstreamer-$(GSTREAMER_VERSION) \
	-Lgstreamer-video-$(GSTREAMER_VERSION) \
	-Lgstreamer-rtsp-server-$(GSTREAMER_VERSION) \
	-lgstapp-1.0 \
	-L/usr/local/cuda/lib64/ -lcudart \
	-Wl,-rpath,$(LIB_INSTALL_DIR)

ifeq ($(BUILD_WEBRTC),true)
LIBS+= -Lgstreamer-sdp-$(GSTREAMER_SDP_VERSION) \
	-Lgstreamer-webrtc-$(GSTREAMER_WEBRTC_VERSION) \
	-Llibsoup-$(LIBSOUP_VERSION) \
	-Ljson-glib-$(JSON_GLIB_VERSION)	
endif

ifeq ($(BUILD_WITH_FFMPEG),true)
LIBS+= -lavformat \
	-lavcodec \
	-lavutil \
	-lswscale \
	-lz \
	-lpthread \
	-lswresample
endif

PKGS:= gstreamer-$(GSTREAMER_VERSION) \
	gstreamer-video-$(GSTREAMER_VERSION) \
	gstreamer-rtsp-server-$(GSTREAMER_VERSION) \
	x11

ifeq ($(BUILD_WEBRTC),true)
PKGS+= gstreamer-sdp-$(GSTREAMER_SDP_VERSION) \
	gstreamer-webrtc-$(GSTREAMER_WEBRTC_VERSION) \
	libsoup-$(LIBSOUP_VERSION) \
	json-glib-$(JSON_GLIB_VERSION)
endif

CFLAGS+= `pkg-config --cflags $(PKGS)`

LIBS+= `pkg-config --libs $(PKGS)`

ifeq ($(BUILD_WITH_OPENCV),true)
PKGS+= opencv4
endif

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
	@echo ----------------------------------------------------------------------
	@echo -- NOTICE: '"make lib"' has been replaced with '"sudo make install"'
	@echo ----------------------------------------------------------------------
	
install:
	if [ ! -d "/tmp/.dsl" ]; then \
		mkdir -p /tmp/.dsl; \
		chmod -R a+rwX /tmp/.dsl; \
	fi
	ar rcs $(LIB).a $(OBJS)
	ar dv $(LIB).a DslCatch.o $(TEST_OBJS)
	$(CXX) -shared $(OBJS) -o $(LIB).so $(LIBS)
	cp -f $(LIB).so /usr/local/lib
	if [ ! -d $(USER_SITE) ]; then \
		mkdir -p $(USER_SITE); \
	fi
	cp -rf ./dsl.py $(USER_SITE)
	
debug_lib:
	$(CXX) -shared $(OBJS) -o $(LIB).so $(LIBS) 
	cp $(LIB).so examples/python/

clean:
	rm -rf $(OBJS) $(APP) $(LIB).a $(LIB).so $(PCH_OUT)
