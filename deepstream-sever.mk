##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Debug
ProjectName            :=deepstream-sever
ConfigurationName      :=Debug
WorkspacePath          :=/home/rjhowell44/Public/CanAmMex/deepstream-sever
ProjectPath            :=/home/rjhowell44/Public/CanAmMex/deepstream-sever
IntermediateDirectory  :=./Debug
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=Robert
Date                   :=17/09/19
CodeLitePath           :=/home/rjhowell44/.codelite
LinkerName             :=/usr/bin/aarch64-linux-gnu-g++
SharedObjectLinkerName :=/usr/bin/aarch64-linux-gnu-g++ -shared -fPIC
ObjectSuffix           :=.o
DependSuffix           :=.o.d
PreprocessSuffix       :=.i
DebugSwitch            :=-g 
IncludeSwitch          :=-I
LibrarySwitch          :=-l
OutputSwitch           :=-o 
LibraryPathSwitch      :=-L
PreprocessorSwitch     :=-D
SourceSwitch           :=-c 
OutputFile             :=$(IntermediateDirectory)/$(ProjectName)
Preprocessors          :=
ObjectSwitch           :=-o 
ArchiveOutputSwitch    := 
PreprocessOnlySwitch   :=-E
ObjectsFileList        :="deepstream-sever.txt"
PCHCompileFlags        :=
MakeDirCommand         :=mkdir -p
LinkOptions            :=  -O0
IncludePath            :=  $(IncludeSwitch). $(IncludeSwitch). 
IncludePCH             := 
RcIncludePath          := 
Libs                   := 
ArLibs                 :=  
LibPath                := $(LibraryPathSwitch). $(LibraryPathSwitch). $(LibraryPathSwitch)Debug 

##
## Common variables
## AR, CXX, CC, AS, CXXFLAGS and CFLAGS can be overriden using an environment variables
##
AR       := /usr/bin/aarch64-linux-gnu-ar rcu
CXX      := /usr/bin/aarch64-linux-gnu-g++
CC       := /usr/bin/aarch64-linux-gnu-gcc
CXXFLAGS :=  -g -Wall $(Preprocessors)
CFLAGS   :=   $(Preprocessors)
ASFLAGS  := 
AS       := /usr/bin/aarch64-linux-gnu-as


##
## User defined environment variables
##
CodeLiteDir:=/usr/share/codelite
Srcs=src/DsdDriver.cpp src/DsdLog4cxx.cpp src/DsdConfig.cpp src/DsdPipeline.cpp src/DsdMain.cpp 

Objects0=$(IntermediateDirectory)/src_DsdDriver.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_DsdLog4cxx.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_DsdConfig.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_DsdPipeline.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_DsdMain.cpp$(ObjectSuffix) 



Objects=$(Objects0) 

##
## Main Build Targets 
##
.PHONY: all clean PreBuild PrePreBuild PostBuild MakeIntermediateDirs
all: $(OutputFile)

$(OutputFile): $(IntermediateDirectory)/.d $(Objects) 
	@$(MakeDirCommand) $(@D)
	@echo "" > $(IntermediateDirectory)/.d
	@echo $(Objects0)  > $(ObjectsFileList)
	$(LinkerName) $(OutputSwitch)$(OutputFile) @$(ObjectsFileList) $(LibPath) $(Libs) $(LinkOptions)

MakeIntermediateDirs:
	@test -d ./Debug || $(MakeDirCommand) ./Debug


$(IntermediateDirectory)/.d:
	@test -d ./Debug || $(MakeDirCommand) ./Debug

PreBuild:
##
## Clean
##
clean:
	$(RM) -r ./Debug/


