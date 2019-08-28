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

APP:= ds-server

CC = g++

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=4.0


SRC_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/
INC_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/includes/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

SRCS:= $(wildcard ./src/*.cpp)
SRCS+= $(wildcard ../../apps-common/src/*.c)

INCS:= $(wildcard ./src/*.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11

OBJS:= $(SRCS:.cpp=.o)

CFLAGS+= -I$(INC_INSTALL_DIR) \
    -I$(SRC_INSTALL_DIR)/apps/apps-common/includes \
    -I/opt/include \
    -DDS_VERSION_MINOR=0 \
    -DDS_VERSION_MAJOR=4 \
    -DDSS_LOGGER_IMP='"DssLog4cxx.h"'

LIBS+= -L$(LIB_INSTALL_DIR) -llog4cxx -laprutil-1 -lapr-1
# LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvds_utils -lm \
#        -lgstrtspserver-1.0 -Wl,-rpath,$(LIB_INSTALL_DIR)

# CFLAGS+= `pkg-config --cflags $(PKGS)`

# LIBS+= `pkg-config --libs $(PKGS)`

all: $(APP)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CC) -o $(APP) $(OBJS) $(LIBS)

clean:
	rm -rf $(OBJS) $(APP)
