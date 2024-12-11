/*
The MIT License

Copyright (c) 2024, Prominence AI, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in-
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "catch.hpp"
#include "DslInferBintr.h"

static std::string primaryAieName("primary-aie");
static std::string inferConfigFile(
    "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-audio/configs/config_infer_audio_sonyc.txt");
static std::string modelEngineFile(
    "/opt/nvidia/deepstream/deepstream/samples/models/SONYC_Audio_Classifier/sonyc_audio_classify.onnx_b2_gpu0_fp32.engine");

static uint hopSize(110250);
static uint frameSize(441000);
static std::string transform("melsdb,fft_length=2560,hop_size=692,dsp_window=hann,num_mels=128,sample_rate=44100,p2db_ref=(float)1.0,p2db_min_power=(float)0.0,p2db_top_db=(float)80.0");


using namespace DSL;

SCENARIO( "A new PrimaryAieBintr is created correctly",  "[PrimaryAieBintr]" )
{
    GIVEN( "Attributes for a new PrimaryAieBintr" ) 
    {
        
        WHEN( "A new PrimaryAieBintr is created" )
        {
            DSL_PRIMARY_AIE_PTR pPrimaryAieBintr= 
                DSL_PRIMARY_AIE_NEW(primaryAieName.c_str(), 
                inferConfigFile.c_str(), modelEngineFile.c_str(),
                frameSize, hopSize, transform.c_str());

            THEN( "The PrimaryAieBintr's memebers are setup and returned correctly" )
            {
                std::string returnedInferConfigFile = pPrimaryAieBintr->GetInferConfigFile();
                std::string returnedModelEngineFile = pPrimaryAieBintr->GetModelEngineFile();
                REQUIRE( returnedInferConfigFile == inferConfigFile );
                REQUIRE( returnedModelEngineFile == modelEngineFile );
                
                REQUIRE( pPrimaryAieBintr->GetMediaType() == DSL_MEDIA_TYPE_AUDIO_ONLY );
                REQUIRE( pPrimaryAieBintr->GetFrameSize() == frameSize );
                REQUIRE( pPrimaryAieBintr->GetHopSize() == hopSize );

                std::string retTransform = pPrimaryAieBintr->GetTransform();
                REQUIRE( retTransform == transform );

                REQUIRE( pPrimaryAieBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A new PrimaryAieBintr can link and unlink all",  "[PrimaryAieBintr]" )
{
    GIVEN( "A new PrimaryAieBintr" ) 
    {
        DSL_PRIMARY_AIE_PTR pPrimaryAieBintr= 
            DSL_PRIMARY_AIE_NEW(primaryAieName.c_str(), 
            inferConfigFile.c_str(), modelEngineFile.c_str(),
            frameSize, hopSize, transform.c_str());

        WHEN( "The PrimaryAieBintr is linked" )
        {
            REQUIRE( pPrimaryAieBintr->SetBatchSize(1) == true );

            REQUIRE( pPrimaryAieBintr->LinkAll() == true );
            REQUIRE( pPrimaryAieBintr->IsLinked() == true );

            THEN( "The PrimaryAieBintr's memebers are setup and returned correctly" )
            {
                pPrimaryAieBintr->UnlinkAll();
                REQUIRE( pPrimaryAieBintr->IsLinked() == false );
            }
        }
    }
}
