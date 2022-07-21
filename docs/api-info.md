# Info API Reference
The Informational Services API provides runtime access to the current DSL version by calling [dsl_info_version_get](#dsl_info_version_get) and GPU type by calling [dsl_info_gpu_type_get](#dsl_info_gpu_type_get).

The standard output stream (stdout) can be redirected to a file by calling [dsl_info_stdout_redirect](#dsl_info_stdout_redirect) or [dsl_info_stdout_redirect_with_ts](#dsl_info_stdout_redirect_with_ts). The current filename, if redirection is active, can be queried by calling [dsl_info_stdout_get](#dsl_info_stdout_get). The standard output stream can be restored by calling [dsl_info_stdout_restore](#dsl_info_stdout_restore)

Applications can control the GStreamer debug log level - by calling [dsl_info_log_level_set](#dsl_info_log_level_set) - and the debug log file - by calling [dsl_info_log_file_set](#dsl_info_log_file_set) or [dsl_info_log_file_set_with_ts](#dsl_info_log_file_set). The `level` and `file_path` values can be queried by calling [dsl_info_log_level_get](#dsl_info_log_level_get) and [dsl_info_log_file_get](#dsl_info_log_file_get) respectively. The default logging function can be restored by calling [dsl_info_log_function_restore](#dsl_info_log_file_set).

---
## Info API
**Methods**
* [dsl_info_version_get](#dsl_info_version_get)
* [dsl_info_gpu_type_get](#dsl_info_gpu_type_get)
* [dsl_info_stdout_get](#dsl_info_stdout_get)
* [dsl_info_stdout_redirect](#dsl_info_stdout_redirect)
* [dsl_info_stdout_redirect_with_ts](#dsl_info_stdout_redirect_with_ts)
* [dsl_info_stdout_restore](#dsl_info_stdout_restore)
* [dsl_info_log_level_get](#dsl_info_log_level_get)
* [dsl_info_log_level_set](#dsl_info_log_level_set)
* [dsl_info_log_file_get](#dsl_info_log_file_get)
* [dsl_info_log_file_set](#dsl_info_log_file_set)
* [dsl_info_log_file_set_with_ts](#dsl_info_log_file_set)
* [dsl_info_log_function_restore](#dsl_info_log_file_set)

---

## Return Values
The following return codes are used by the DSL Info API
```C++
#define DSL_RESULT_SUCCESS                                          0x00000000
#define DSL_RESULT_FAILURE                                          0x00000001
#define DSL_RESULT_INVALID_INPUT_PARAM                              0x00000005
#define DSL_RESULT_THREW_EXCEPTION                                  0x00000006
```

<br>

## GPU Type Values
The following GPU Type values are used by the DSL Info API
```C
#define DSL_GPU_TYPE_INTEGRATED                                     0
#define DSL_GPU_TYPE_DISCRETE                                       1
```

<br>

## File Open Modes
The following file open modes are used by the DSL Info API
```c
#define DSL_WRITE_MODE_APPEND                                       0
#define DSL_WRITE_MODE_TRUNCATE                                     1
```

<br>
 
---

## Methods
### *dsl_info_version_get*
```C++
const wchar_t* dsl_info_version_get();
```
This serverice returns the current version of DSL.

**Returns**
* A constant string representation of the current DSL release.

**Python Example**
```Python
print('Current DSL release =', dsl_info_version_get())
```

<br>

### *dsl_info_gpu_type_get*
```C++
uint dsl_info_gpu_type_get(uint gpu_id);
```
This services returns the GPU type for a given GPU.

**Parameters**
* `gpu_id` - [in] unique identifier for the GPU to query.

**Returns**
* On of the [GPU Type Values](gpu-type-values) defined above.

**Python Example**
```Python
if dsl_info_gpu_type_get(gpu_id=0) == DSL_GPU_TYPE_DISCRETE:
   # handle platform specific code...
```

<br>

### *dsl_info_stdout_get*
```C++
DslReturnType dsl_info_stdout_get(const wchar_t** file_path);
```
This service gets the current setting for where stdout is directed. The service returns `"console"` unless stdout has been redirected to a log file.

**Parameters**
* `file_path` - [out] absolute file path specification or `"console"`.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
stdout_file_path = dsl_info_stdout_get()
```

<br>

### *dsl_info_stdout_redirect*
```C++
DslReturnType dsl_info_stdout_redirect(const wchar_t* file_path, uint mode);
```
This service redirects all data streamed to stdout to a specified file. The current running log file will be saved if stdout is already redirected. 

Note: stdout redirection does not include the GStreamer debug logs even when sent to the console, nor does it include print statements from a calling Python application. 

**Important notes**
* This service appends a `.log` extension to `file_path` on file open.
* All folders specified in `file_path` must exist or this service will fail.

**Parameters**
* `file_path` - [in] absolute or relative file path specification for the output log file.
* `mode` - [in] file open mode if the file already exists. One of the [File Open Modes](#file-open-modes) defined above.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_info_stdout_redirect('/tmp/.dsl/stdout', DSL_WRITE_MODE_APPEND)
```
<br>

### *dsl_info_stdout_redirect_with_ts*
```C++
DslReturnType dsl_info_stdout_redirect_with_ts(const wchar_t* file_path);
```
This service redirects all data streamed to stdout to a specified file. The current running log file will be saved if stdout is already redirected. The file is opened with the current date and time appended to `file_path`

Note: stdout redirection does not include the GStreamer debug logs even when sent to the console, nor does it include print statements from a calling Python application. 

**Important notes**
* This service appends a `<%Y%m%d-%H%M%S>.log` timestamp and extension to `file_path` on file open.
* All folders specified in `file_path` must exist or this service will fail.

**Parameters**
* `file_path` - [in] absolute or relative file path specification for the output log file.

**Returns**
* `DSL_RESULT_SUCCESS` on successful redirect. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_info_stdout_redirect_with_ts('/tmp/.dsl/stdout')
```
<br>

### *dsl_info_stdout_restore*
```C++
DslReturnType dsl_info_stdout_restore();
```
This services restores stdout from file redirection.

**Returns**
* `DSL_RESULT_SUCCESS` on successful restore. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_info_stdout_restore()
```
<br>

### *dsl_info_log_level_get*
```C++
DslReturnType dsl_info_log_level_get(const wchar_t** level);
```
This service gets the current GST debug log level if previously set with the environment variable `GST_DEBUG` or with a call to [dsl_info_log_level_set](#dsl_info_log_level_set).

**Parameters**
* `level` - [out] current level string defining one or more debug group/level pairs prefixed with an optional global default. Empty string if undefined.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
current_debug_log_level = dsl_info_log_level_get()
```
<br>

### *dsl_info_log_level_set*
```C++
DslReturnType dsl_info_log_level_set(const wchar_t* level);
```
This service sets the GST debug log level. The call will override the current value of the `GST_DEBUG` environment variable.

**Parameters**
* `level` - [in] new level (string) defining one or more debug group/level pairs prefixed with optional global default.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_info_log_level_set('1,DSL:2,MY-CATEGORY:5')
```
<br>

### *dsl_info_log_file_get*
```C++
DslReturnType dsl_info_log_file_get(const wchar_t** file_path);
```
This service gets the current GST debug log file if previously set with the environment variable `GST_DEBUG_FILE` or with a call to [dsl_info_log_file_set](#dsl_info_log_file_set) or [dsl_info_log_file_set_with_ts. ](#dsl_info_log_file_set_with_ts).

**Parameters**
* `file_path` - [out] absolute file path specification or empty string if undefined.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
debug_log_file_path = dsl_info_log_file_get()
```
<br>

### *dsl_info_log_file_set*
```C++
DslReturnType dsl_info_log_file_set(const wchar_t* file_path, uint mode);
```
This service sets the GST debug log file. The call will override the current value of the `DSL_DEBUG_FILE` environment variable. The current running log file will be saved if  in progress. The file can be opened for append or truncated if found.

**Important notes**
* This services appends a `.log` extension to `file_path` on file open.
* All folders specified in `file_path` must exist or this service will fail.

**Parameters**
* `file_path` - [in] absolute or relative file path specification for the debug log file.
* `mode` - [in] file open mode if the file already exists. One of the [File Open Modes](#file-open-modes) defined above.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_info_log_file_set('/tmp/.dsl/debug', DSL_WRITE_MODE_APPEND)
```
<br>

### *dsl_info_log_file_set_with_ts*
```C++
DslReturnType dsl_info_log_file_set_with_ts(const wchar_t* file_path);
```
This service sets the GST debug log file. The call will override the current value of the `DSL_DEBUG_FILE` environment variable. The current running log file will be saved if in progress. The new file will be opened with the current date and time appended to `file_path`

**Important notes**
* This services appends a `<%Y%m%d-%H%M%S>.log` timestamp and extension to `file_path` on file open.
* Any folders specified in `file_path` must exist or this service will fail.

**Parameters**
* `file_path` - [in] absolute or relative file path specification for the debug log file.

**Returns**
* `DSL_RESULT_SUCCESS` on successful redirect. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_info_log_file_set_with_ts('/tmp/.dsl/debug')
```

<br>

### *dsl_info_log_function_restore*
```C++
DslReturnType dsl_info_log_function_restore();
```
This service restores the original default GST log function which will write logs to `GST_DEBUG_FILE` if set, or stdout otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful restore. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_info_log_function_restore()
```
<br>

---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Player](/docs/api-player.md)
* [Source](/docs/api-source.md)
* [Tap](/docs/api-tap.md)
* [Dewarper](/docs/api-dewarper.md)
* [Preprocessor](/docs/api-preproc.md)
* [Inference Engine and Server](/docs/api-infer.md)
* [Tracker](/docs/api-tracker.md)
* [Segmentation Visualizer](/docs/api-segvisual.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* **Info API**
