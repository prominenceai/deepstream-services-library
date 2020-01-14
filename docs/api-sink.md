# Sink API

## Tiled Display API
* [dsl_tiler_new](#dsl_tiler_new)

## Return Values
The following return codes are used by the Sink API
```C++
#define DSL_RESULT_SINK_NAME_NOT_UNIQUE                             0x00040001
#define DSL_RESULT_SINK_NAME_NOT_FOUND                              0x00040002
#define DSL_RESULT_SINK_NAME_BAD_FORMAT                             0x00040003
#define DSL_RESULT_SINK_THREW_EXCEPTION                             0x00040004
#define DSL_RESULT_SINK_FILE_PATH_NOT_FOUND                         0x00040005
#define DSL_RESULT_SINK_IS_IN_USE                                   0x00040007
#define DSL_RESULT_SINK_SET_FAILED                                  0x00040008
```
## Codec Types
The following codec type codes are used by the Sink API
```C++
#define DSL_CODEC_H264                                              0
#define DSL_CODEC_H265                                              1
#define DSL_CODEC_MPEG4                                             2
```
<br>

## Constructors
### *dsl_sink_overlay_new*
```C++
DslReturnType dsl_sink_overlay_new(const wchar_t* name, 
    uint offsetX, uint offsetY, uint width, uint height);
```
The constructor creates a uniquely named Overlay Sink with given offsets and dimensions. Construction will fail if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the Overlay Sink to create.
* `offsetX` - [in] offset in the X direction from the upper left corner in pixels
* `offsetY` - [in] offset in the Y direction 
* `width` - [in] width of the Overlay Sink in pixels
* `height` - [in] height of the Overlay Sink in pixels

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_sink_window_new*
```C++
DslReturnType dsl_sink_window_new(const wchar_t* name, 
    uint offsetX, uint offsetY, uint width, uint height);
```
The constructor creates a uniquely named Window Sink with given offsets and dimensions. Construction will fail if the name is currently in use. Window Sinks are used render video to an XWindow's Display. See [Pipeline XWindow Support](api-pipeline.md#pipeline-xwindow-support) for more information.

**Parameters**
* `name` - [in] unique name for the Window Sink to create.
* `offsetX` - [in] offset in the X direction from the upper left corner in pixels
* `offsetY` - [in] offset in the Y direction from the upper left corner in pixels
* `width` - [in] width of the Overlay Sink in pixels
* `height` - [in] height of the Overlay Sink in pixels

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_sink_file_new*
```C++
DslReturnType dsl_sink_file_new(const wchar_t* name, const wchar_t* filepath, 
     uint codec, uint muxer, uint bit_rate, uint interval);
```
The constructor creates a uniquely named File Sink. Construction will fail if the name is currently in use. There are three Codec formats support - `H.264`, `H.265`, and `MPEG`.-
