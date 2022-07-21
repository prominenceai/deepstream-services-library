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


APP:= dsl-test-app.exe
LIB:= libdsl

CXX = g++

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
USER_SITE = "`python3 -m site --user-site`"

CXX_VERSION:=c++17
DSL_VERSION:='L"v0.25.alpha"'
GLIB_VERSION:=2.0
GSTREAMER_VERSION:=1.0
GSTREAMER_SUB_VERSION:=14
GSTREAMER_SDP_VERSION:=1.0
GSTREAMER_WEBRTC_VERSION:=1.0
LIBSOUP_VERSION:=2.4
JSON_GLIB_VERSION:=1.0

SRC_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/sources
INC_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/sources/includes
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib

SRCS+= $(wildcard ./src/*.cpp)
SRCS+= $(wildcard ./src/thirdparty/*.cpp)
SRCS+= $(wildcard ./test/*.cpp)
SRCS+= $(wildcard ./test/api/*.cpp)
SRCS+= $(wildcard ./test/unit/*.cpp)

INCS+= $(wildcard ./src/*.h)
INCS+= $(wildcard ./src/thirdparty/*.h)
INCS+= $(wildcard ./test/*.hpp)

ifeq ($(GSTREAMER_SUB_VERSION),18)
SRCS+= $(wildcard ./src/webrtc/*.cpp)
SRCS+= $(wildcard ./test/webrtc/*.cpp)
INCS+= $(wildcard ./src/webrtc/*.h)
endif

TEST_OBJS+= $(wildcard ./test/api/*.o)
TEST_OBJS+= $(wildcard ./test/unit/*.o)
ifeq ($(GSTREAMER_SUB_VERSION),18)
TEST_OBJS+= $(wildcard ./test/webrtc/*.o)
endif

OBJS:= $(SRCS:.c=.o)
OBJS:= $(OBJS:.cpp=.o)

CFLAGS+= -I$(INC_INSTALL_DIR) \
	-std=$(CXX_VERSION) \
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
	-DGSTREAMER_SUB_VERSION=$(GSTREAMER_SUB_VERSION) \
	-DBUILD_MESSAGE_SINK=$(BUILD_MESSAGE_SINK) \
	-DNVDS_DCF_LIB='"$(LIB_INSTALL_DIR)/libnvds_nvdcf.so"' \
	-DNVDS_KLT_LIB='"$(LIB_INSTALL_DIR)/libnvds_mot_klt.so"' \
	-DNVDS_IOU_LIB='"$(LIB_INSTALL_DIR)/libnvds_mot_iou.so"' \
	-DNVDS_MOT_LIB='"$(LIB_INSTALL_DIR)/libnvds_nvmultiobjecttracker.so"' \
	-DNVDS_AMQP_PROTO_LIB='L"$(LIB_INSTALL_DIR)/libnvds_amqp_proto.so"' \
	-DNVDS_AZURE_PROTO_LIB='L"$(LIB_INSTALL_DIR)/libnvds_azure_proto.so"' \
	-DNVDS_AZURE_EDGE_PROTO_LIB='L"$(LIB_INSTALL_DIR)/libnvds_azure_edge_proto"' \
	-DNVDS_KAFKA_PROTO_LIB='L"$(LIB_INSTALL_DIR)/libnvds_kafka_proto.so"' \
	-DNVDS_REDIS_PROTO_LIB='L"$(LIB_INSTALL_DIR)/libnvds_redis_proto.so"' \
    -fPIC 

ifeq ($(GSTREAMER_SUB_VERSION),18)
CFLAGS+= -I/usr/include/libsoup-$(LIBSOUP_VERSION) \
	-I/usr/include/json-glib-$(JSON_GLIB_VERSION) \
	-I./src/webrtc
endif	

CFLAGS += `geos-config --cflags`	

LIBS+= -L$(LIB_INSTALL_DIR) \
	-L/usr/local/lib \
	-lopencv_core \
	-lopencv_imgproc \
	-lopencv_highgui \
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
	-L/usr/local/cuda/lib64/ -lcudart \
	-Wl,-rpath,$(LIB_INSTALL_DIR)

ifeq ($(GSTREAMER_SUB_VERSION),18)
LIBS+= -Lgstreamer-sdp-$(GSTREAMER_SDP_VERSION) \
	-Lgstreamer-webrtc-$(GSTREAMER_WEBRTC_VERSION) \
	-Llibsoup-$(LIBSOUP_VERSION) \
	-Ljson-glib-$(JSON_GLIB_VERSION)	
endif

PKGS:= gstreamer-$(GSTREAMER_VERSION) \
	gstreamer-video-$(GSTREAMER_VERSION) \
	gstreamer-rtsp-server-$(GSTREAMER_VERSION) \
	x11 \
	opencv4

ifeq ($(GSTREAMER_SUB_VERSION),18)
PKGS+= gstreamer-sdp-$(GSTREAMER_SDP_VERSION) \
	gstreamer-webrtc-$(GSTREAMER_WEBRTC_VERSION) \
	libsoup-$(LIBSOUP_VERSION) \
	json-glib-$(JSON_GLIB_VERSION)
endif

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
