# ODE Action Services API
#### ODE Action Construction and Destruction
#### Actions on Actions
#### Actions on Areas
#### Actions on Triggers
#### Actions on Pipelines
#### Actions on Meta Data

## ODE Services API
**Constructors:**
* [dsl_ode_action_action_add_new](#dsl_ode_action_action_add_new)
* [dsl_ode_action_action_disable_new](#dsl_ode_action_action_disable_new)
* [dsl_ode_action_action_enable_new](#dsl_ode_action_action_enable_new)
* [dsl_ode_action_action_remove_new](#dsl_ode_action_action_remove_new)
* [dsl_ode_action_area_add_new](#dsl_ode_action_area_add_new)
* [dsl_ode_action_area_remove_new](#dsl_ode_action_area_remove_new)
* [dsl_ode_action_callback_new](#dsl_ode_action_callback_new)
* [dsl_ode_action_capture_frame_new](#dsl_ode_action_capture_frame_new)
* [dsl_ode_action_capture_object_new](#dsl_ode_action_capture_object_new)
* [dsl_ode_action_display_new](#dsl_ode_action_display_new)
* [dsl_ode_action_dump_new](#dsl_ode_action_dump_new)
* [dsl_ode_action_fill_new](#dsl_ode_action_fill_new)
* [dsl_ode_action_hide_new](#dsl_ode_action_hide_new)
* [dsl_ode_action_kitti_new](#dsl_ode_action_kitti_new)
* [dsl_ode_action_log_new](#dsl_ode_action_log_new)
* [dsl_ode_action_pause_new](#dsl_ode_action_pause_new)
* [dsl_ode_action_print_new](#dsl_ode_action_print_new)
* [dsl_ode_action_redact_new](#dsl_ode_action_redact_new)
* [dsl_ode_action_sink_add_new](#dsl_ode_action_sink_add_new)
* [dsl_ode_action_sink_remove_new](#dsl_ode_action_sink_remove_new)
* [dsl_ode_action_source_add_new](#dsl_ode_action_source_add_new)
* [dsl_ode_action_source_remove_new](#dsl_ode_action_source_remove_new)
* [dsl_ode_action_trigger_add_new](#dsl_ode_action_trigger_add_new)
* [dsl_ode_action_trigger_disable_new](#dsl_ode_action_trigger_disable_new)
* [dsl_ode_action_trigger_enable_new](#dsl_ode_action_trigger_enable_new)
* [dsl_ode_action_trigger_remove_new](#dsl_ode_action_trigger_remove_new)

**Destructors:**
* [dsl_ode_action_delete](#dsl_ode_action_delete)
* [dsl_ode_action_delete_many](#dsl_ode_action_delete_many)
* [dsl_ode_action_delete_all](#dsl_ode_action_delete_all)

**Methods:**
* [dsl_ode_action_enabled_get](#dsl_ode_action_enable_get)
* [dsl_ode_action_enabled_set](#dsl_ode_action_enable_set)
* [dsl_ode_action_list_size](#dsl_ode_action_list_size)
* [dsl_ode_action_list_size](#dsl_ode_action_list_size)

---
## Return Values
The following return codes are used by the OSD Action API
```C++
#define DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE                       0x000F0001
#define DSL_RESULT_ODE_ACTION_NAME_NOT_FOUND                        0x000F0002
#define DSL_RESULT_ODE_ACTION_CAPTURE_TYPE_INVALID                  0x000F0003
#define DSL_RESULT_ODE_ACTION_THREW_EXCEPTION                       0x000F0004
#define DSL_RESULT_ODE_ACTION_IN_USE                                0x000F0005
#define DSL_RESULT_ODE_ACTION_SET_FAILED                            0x000F0006
#define DSL_RESULT_ODE_ACTION_IS_NOT_ACTION                         0x000F0007
#define DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND                   0x000F0008
#define DSL_RESULT_ODE_ACTION_NOT_THE_CORRECT_TYPE                  0x000F0009
```
---
## Constructors
### *dsl_ode_action_action_add_new*
```C++
DslReturnType dsl_ode_action_action_add_new(const wchar_t* name, 
    const wchar_t* trigger, const wchar_t* action);
```
The constructor creates a uniquely named Add Action ODE Action. When invoked, this Action will attempt to add a named ODE Action to a named ODE Trigger. The Action will produce an error message if either the Trigger or Action do not exist.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `trigger` - [in] unique name for the ODE Trigger to add the ODE Action to.
* `action` - [in] unique name for the ODE Action to add to the ODE Trigger.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_action_add_new('my-add-action-action', 'my-trigger', 'my-other-action')
```

<br>

### *dsl_ode_action_action_disable_new*
```C++
DslReturnType dsl_ode_action_action_disable_new(const wchar_t* name, const wchar_t* action);
```
The constructor creates a uniquely named Disable Action ODE Action. When invoked, this Action will attempt to disable a named ODE Action. The Action will produce an error log message if the Action to enable does not exist at the time of invocation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `action` - [in] unique name for the ODE Action to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_action_disable_new('my-disable-action-action', 'my-other-action')
```

<br>

### *dsl_ode_action_action_enable_new*
```C++
DslReturnType dsl_ode_action_action_enable_new(const wchar_t* name, const wchar_t* action);
```
The constructor creates a uniquely named Enable Action ODE Action. When invoked, this Action will attempt to enable a named ODE Action. The Action will produce an error log message if the Action to enable does not exist at the time of invocation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `action` - [in] unique name for the ODE Action to enable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_action_enable_new('my-enable-action-action', 'my-other-action')
```

<br>

### *dsl_ode_action_action_remove_new*
```C++
DslReturnType dsl_ode_action_action_remove_new(const wchar_t* name, 
    const wchar_t* trigger, const wchar_t* action);
```
The constructor creates a uniquely named Remove Action ODE Action. When invoked, this Action will attempt to remove a named ODE Action to a named ODE Trigger. The Action will produce an error log message if either the Trigger or Action do not exist at the time of invocation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `trigger` - [in] unique name for the ODE Trigger to remove the ODE Action from.
* `action` - [in] unique name for the ODE Action to remove from the ODE Trigger.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_action_remove_new('my-remove-action-action', 'my-trigger', 'my-other-action')
```

<br>

### *dsl_ode_action_area_add_new*
```C++
DslReturnType dsl_ode_action_area_add_new(const wchar_t* name, 
    const wchar_t* trigger, const wchar_t* area);
```
The constructor creates a uniquely named Add Area ODE Action. When invoked, this Action will attempt to add a named ODE Area to a named ODE Trigger. The Action will produce an error message if either the Trigger or Area do not exist at the time of invocation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `trigger` - [in] unique name for the ODE Trigger to add the ODE Area to.
* `area` - [in] unique name for the ODE Area to add to the ODE Trigger.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_area_add_new('my-add-area-action', 'my-trigger', 'my-area')
```

<br>

### *dsl_ode_action_area_remove_new*
```C++
DslReturnType dsl_ode_action_area_remove_new(const wchar_t* name, 
    const wchar_t* trigger, const wchar_t* area);
```
The constructor creates a uniquely named Remove Area ODE Action. When invoked, this Action will attempt to remove a named ODE Area from a named ODE Trigger. The Action will produce an error log message if either the Trigger or Area do not exist at the time of invocation. 

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `trigger` - [in] unique name for the ODE Trigger to remove the ODE Area from.
* `area` - [in] unique name for the ODE Area to add to the ODE Trigger.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_area_remove_new('my-remove-area-action', 'my-trigger', 'my-area')
```

<br>

### *dsl_ode_action_callback_new*
```C++
DslReturnType dsl_ode_action_callback_new(const wchar_t* name, 
    dsl_ode_occurrence_handler_cb client_handler, void* client_data);
```
The constructor creates a uniquely named Callback ODE Action. When invoked, this Action will call the Client provided callback function with the Frame Meta and Object Meta that triggered the ODE occurrence. 

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `client_handler` - [in] Function of type `dsl_ode_occurrence_handler_cb` to be called on Action Invocation.
* `client_data` - [in] Opaque pointer to client data returned on callback.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_callback_new('my-callback-action', my_ode_callback, my_data)
```

<br>

### *dsl_ode_action_capture_frame_new*
```C++
DslReturnType dsl_ode_action_capture_frame_new(const wchar_t* name, const wchar_t* outdir);
```
The constructor creates a uniquely named Frame Capture ODE Action. When invoked, this Action will capture the frame that triggered the ODE occurrence to a jpeg image file in the directory specified by `outdir`. The file name will be derived from combining the unique ODE Trigger name and unique ODE occurrence ID. The constructor will return `DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND` if `outdir` is invalid.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `outdir` - [in] absolute or relative path to the output directory to save the image file to

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_capture_frame_new('my-frame-capture-action', './images/frames')
```

<br>

### *dsl_ode_action_capture_object_new*
```C++
DslReturnType dsl_ode_action_capture_object_new(const wchar_t* name, const wchar_t* outdir);
```
The constructor creates a uniquely named Object Capture ODE Action. When invoked, this Action will capture the object -- using its OSD rectangle parameters --  that triggered the ODE occurrence to a jpeg image file in the directory specified by `outdir`. The file name will be derived from combining the unique ODE Trigger name and unique ODE occurrence ID. The constructor will return `DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND` if `outdir` is invalid.

Note: Adding an Object Capture ODE Action to an Absence or Summation Trigger is meaningless and will result in a Non-Operation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `outdir` - [in] absolute or relative path to the output directory to save the image file to

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_capture_object_new('my-object-action', './images/frames')
```

<br>

### *dsl_ode_action_display_new*
```C++
DslReturnType dsl_ode_action_display_new(const wchar_t* name, 
    uint offsetX, uint offsetY, boolean offsetY_with_classId);
```
The constructor creates a uniquely named Display ODE Action. When invoked, this Action writes the ODE Trigger's name and occurrence count as meta data to the current Frame Meta for display by a downstream On-Screen-Display (OSD) component.


**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `offsetX` - [in] offset for the display text in the X direction.
* `offsetY` - [in] offset for the display text in the Y direction.
* `offsetY_with_classId` - [in] if true adds an additional Y offset based on the Class Id of the Trigger invoking the Action. This allows multiple Triggers with different Class Ids to share the same Display Action

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_capture_object_new('my-object-action', 10, 30, True)
```

<br>

### *dsl_ode_action_dump_new*
```C++
DslReturnType dsl_ode_action_dump_new(const wchar_t* name, const wchar_t* pipeline, const wchar_t* filename);
```
The constructor creates a uniquely named Dump ODE Action. When invoked, this Action  dumps a Pipeline's graph to dot file. The GStreamer Pipeline creates a topology graph on each change of state to ready, playing and paused if the debug environment variable GST_DEBUG_DUMP_DOT_DIR is set.

GStreamer will add the .dot suffix and write the file to the directory specified by the environment variable. 

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `pipeline` - [in] unique name of the Pipeline to dump.
* `filename` - [in] name to give the .dot file on dump.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_dump_new('my-pipeline-dump-action', 'my-pipeline', 'my-dumpfile')
```

<br>

### *dsl_ode_action_fill_new*
```C++
DslReturnType dsl_ode_action_fill_new(const wchar_t* name,
    double red, double green, double blue, double alpha);
```
The constructor creates a uniquely named Fill ODE Action. When invoked, this Action will fill the OSD rectangle background color for the Object Meta that triggered the ODE occurrence.

Note: Adding a Fill ODE Action to an Absence or Summation Trigger is meaningless and will result in a Non-Operation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `red` - [in] red value for the RGBA background color [1..0]
* `green` - [in] green value for the RGBA background color [1..0]
* `blue` - [in] blue value for the RGBA background color [1..0]
* `alpah` - [in] alpha value for the RGBA background color [1..0]


**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_fill_new('my-fill-action', 1.0, 0.0, 0.0, 0.0, 0.25)
```

<br>

### *dsl_ode_action_hide_new*
```C++
DslReturnType dsl_ode_action_hide_new(const wchar_t* name, boolean text, boolean border);
```
The constructor creates a uniquely named Hide ODE Action. When invoked, this Action will hide the OSD display test and/or rectangle border for the Object Meta that triggered the ODE occurrence.


**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `text` - [in] if true, the action hides the display text for Object that trigger the ODE occurrence
* `border` - [in] if true, the action hides the rectangle border for Object that triggered the ODE occurrence


**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_hide_new('my-hide-action', True, True)
```

<br>
