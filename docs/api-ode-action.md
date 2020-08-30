# ODE Action Services API
Object Detection Event (ODE) Actions implement a type specific `handle-ode-occurrence` function that gets invoked by an ODE Trigger on ODE Occurrence. The relationship of ODE Trigger to ODE Actions is many-to-many. Multiple ODE Actions can be added to an ODE Trigger and the same ODE Action can be added to multiple ODE Triggers.

#### Actions on Metadata
Several ODE Actions can be created to update the Frame and object Metadata to be used by the [On-Screen-Display](/docs/api-osd.md), the next component in the Pipeline (if added).  by See [dsl_ode_action_fill_area_new](#dsl_ode_action_fill_area_new), [dsl_ode_action_fill_frame_new](#dsl_ode_action_fill_frame_new), [dsl_ode_action_fill_object_new](#dsl_ode_action_fill_object_new), [dsl_ode_action_hide_new](#dsl_ode_action_hide_new), [dsl_ode_action_overlay_frame_new](#dsl_ode_action_overlay_frame_new), and [dsl_ode_action_redact_new](#dsl_ode_action_redact_new)

#### Actions on Record Components
There are two actions for starting a record session, one for the [Record-Sink](/docs/api-sink.md) [dsl_ode_action_sink_record_start_new](#dsl_ode_action_sink_record_start_new) and the other for the [Record-Tap](/docs/api-tap.md) [dsl_ode_action_tap_record_start_new](#dsl_ode_action_tap_record_start_new)

#### Actions on Actions
Actions can be created to Disable other Actions on invocation. See [dsl_ode_action_action_disable_new](#dsl_ode_action_action_disable_new) and [dsl_ode_action_action_enable_new](#dsl_ode_action_action_enable_new). 

#### Actions with ODE Occurrence Data
Actions performed with the ODE occurrence data include  [dsl_ode_action_callback_new](#dsl_ode_action_callback_new), [dsl_ode_action_display_new](#dsl_ode_action_display_new), [dsl_ode_action_log_new](#dsl_ode_action_log_new), and [dsl_ode_action_print_new](#dsl_ode_action_print_new)

#### Actions on Areas
Actions can be used to Add and Remove Areas to/from a Trigger on invocation. See [dsl_ode_action_area_add_new](#dsl_ode_action_area_add_new) and [dsl_ode_action_area_remove_new](#dsl_ode_action_area_remove_new). 

#### Actions on Triggers
Actions can be created to Disable, Enable or Reset a Trigger on invocation. See [dsl_ode_action_trigger_reset_new](#dsl_ode_action_trigger_reset_new), [dsl_ode_action_trigger_disable_new](#dsl_ode_action_trigger_disable_new), and [dsl_ode_action_trigger_enable_new](#dsl_ode_action_trigger_enable_new)

#### Actions on Pipelines
There are a number of Actions that dynamically update the state or components in a Pipeline. [dsl_ode_action_pause_new](#dsl_ode_action_pause_new), [dsl_ode_action_sink_add_new](#dsl_ode_action_sink_add_new), [dsl_ode_action_sink_remove_new](#dsl_ode_action_sink_remove_new), [dsl_ode_action_sink_record_start_new](#dsl_ode_action_sink_record_start_new), [dsl_ode_action_source_add_new](#dsl_ode_action_source_add_new), [dsl_ode_action_source_remove_new](#dsl_ode_action_source_remove_new), and 

#### ODE Action Construction and Destruction
ODE Actions are created by calling one of type specific [constructors](#ode-action-api) defined below. Each constructor must have a unique name and using a duplicate name will fail with a result of `DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE`. Once created, all Actions are deleted by calling [dsl_ode_action_delete](#dsl_ode_action_delete),
[dsl_ode_action_delete_many](#dsl_ode_action_delete_many), or [dsl_ode_action_delete_all](#dsl_ode_action_delete_all). Attempting to delete an Action in-use by a Trigger will fail with a result of `DSL_RESULT_ODE_ACTION_IN_USE`

#### Adding/Removing Actions
ODE Actions are added to an ODE Trigger by calling [dsl_ode_trigger_action_add](docs/api-ode-trigger#dsl_ode_trigger_action_add) and [dsl_ode_trigger_action_add_many](docs/api-ode-trigger#dsl_ode_trigger_action_add_many) and removed with [dsl_ode_trigger_action_remove](docs/api-ode-trigger#dsl_ode_trigger_action_remove), [dsl_ode_trigger_action_remove_many](docs/api-ode-traigger#dsl_ode_trigger_action_remove_many), and [dsl_ode_trigger_action_remove_all](docs/api-ode-trigger#dsl_ode_trigger_action_remove_all).


## ODE Action API
**Constructors:**
* [dsl_ode_action_action_disable_new](#dsl_ode_action_action_disable_new)
* [dsl_ode_action_action_enable_new](#dsl_ode_action_action_enable_new)
* [dsl_ode_action_area_add_new](#dsl_ode_action_area_add_new)
* [dsl_ode_action_area_remove_new](#dsl_ode_action_area_remove_new)
* [dsl_ode_action_custom_new](#dsl_ode_action_custom_new)
* [dsl_ode_action_capture_area_new](#dsl_ode_action_capture_area_new)
* [dsl_ode_action_capture_frame_new](#dsl_ode_action_capture_frame_new)
* [dsl_ode_action_capture_object_new](#dsl_ode_action_capture_object_new)
* [dsl_ode_action_display_new](#dsl_ode_action_display_new)
* [dsl_ode_action_display_meta_add_new](#dsl_ode_action_display_meta_add_new)
* [dsl_ode_action_display_meta_add_many_new](#dsl_ode_action_display_meta_add_many_new)
* [dsl_ode_action_fill_frame_new](#dsl_ode_action_fill_frame_new)
* [dsl_ode_action_fill_object_new](#dsl_ode_action_fill_object_new)
* [dsl_ode_action_handler_disable_new](#dsl_ode_action_handler_disable_new)
* [dsl_ode_action_hide_new](#dsl_ode_action_hide_new)
* [dsl_ode_action_log_new](#dsl_ode_action_log_new)
* [dsl_ode_action_pause_new](#dsl_ode_action_pause_new)
* [dsl_ode_action_print_new](#dsl_ode_action_print_new)
* [dsl_ode_action_redact_new](#dsl_ode_action_redact_new)
* [dsl_ode_action_sink_add_new](#dsl_ode_action_sink_add_new)
* [dsl_ode_action_sink_remove_new](#dsl_ode_action_sink_remove_new)
* [dsl_ode_action_sink_record_start_new](#dsl_ode_action_sink_record_start_new)
* [dsl_ode_action_source_add_new](#dsl_ode_action_source_add_new)
* [dsl_ode_action_source_remove_new](#dsl_ode_action_source_remove_new)
* [dsl_ode_action_tap_record_start_new](#dsl_ode_action_tap_record_start_new)
* [dsl_ode_action_tiler_source_show_new](#dsl_ode_action_tiler_source_show_new)
* [dsl_ode_action_trigger_reset_new](#dsl_ode_action_trigger_reset_new)
* [dsl_ode_action_trigger_disable_new](#dsl_ode_action_trigger_disable_new)
* [dsl_ode_action_trigger_enable_new](#dsl_ode_action_trigger_enable_new)

**Destructors:**
* [dsl_ode_action_delete](#dsl_ode_action_delete)
* [dsl_ode_action_delete_many](#dsl_ode_action_delete_many)
* [dsl_ode_action_delete_all](#dsl_ode_action_delete_all)

**Methods:**
* [dsl_ode_action_enabled_get](#dsl_ode_action_enabled_get)
* [dsl_ode_action_enabled_set](#dsl_ode_action_enabled_set)
* [dsl_ode_action_list_size](#dsl_ode_action_list_size)

---
## Return Values
The following return codes are used by the OSD Action API
```C++
#define DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE                       0x000F0001
#define DSL_RESULT_ODE_ACTION_NAME_NOT_FOUND                        0x000F0002
#define DSL_RESULT_ODE_ACTION_CAPTURE_TYPE_INVALID                  0x000F0003
#define DSL_RESULT_ODE_ACTION_THREW_EXCEPTION                       0x000F0004
#define DSL_RESULT_ODE_ACTION_IN_USE                                0x000F0005
#define DSL_RESULT_ODE_ACTION_SET_FAILED                            0x000F0006
#define DSL_RESULT_ODE_ACTION_IS_NOT_ACTION                         0x000F0007
#define DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND                   0x000F0008
#define DSL_RESULT_ODE_ACTION_NOT_THE_CORRECT_TYPE                  0x000F0009
```
---
## Constructors
### *dsl_ode_action_action_add_new*
```C++
DslReturnType dsl_ode_action_action_add_new(const wchar_t* name, 
    const wchar_t* trigger, const wchar_t* action);
```
The constructor creates a uniquely named **Add Action** ODE Action. When invoked, this Action will attempt to add a named ODE Action to a named ODE Trigger. The Action will produce an error log message if either the Trigger or Action do not exist.

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
The constructor creates a uniquely named **Disable Action** ODE Action. When invoked, this Action will attempt to disable a named ODE Action. The Action will produce an error log message if the Action to disable does not exist at the time of invocation.

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
The constructor creates a uniquely named **Enable Action** ODE Action. When invoked, this Action will attempt to enable a named ODE Action. The Action will produce an error log message if the Action to enable does not exist at the time of invocation.

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

### *dsl_ode_action_area_add_new*
```C++
DslReturnType dsl_ode_action_area_add_new(const wchar_t* name, 
    const wchar_t* trigger, const wchar_t* area);
```
The constructor creates a uniquely named **Add Area** ODE Action. When invoked, this Action will attempt to add a named ODE Area to a named ODE Trigger. The Action will produce an error message if either the Trigger or Area do not exist at the time of invocation.

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
The constructor creates a uniquely named **Remove Area** ODE Action. When invoked, this Action will attempt to remove a named ODE Area from a named ODE Trigger. The Action will produce an error log message if either the Trigger or Area do not exist at the time of invocation. 

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `trigger` - [in] unique name for the ODE Trigger to remove the ODE Area from.
* `area` - [in] unique name for the ODE Area to remove from the ODE Trigger.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_area_remove_new('my-remove-area-action', 'my-trigger', 'my-area')
```

<br>

### *dsl_ode_action_custom_new*
```C++
DslReturnType dsl_ode_action_custom_new(const wchar_t* name, 
    dsl_ode_occurrence_handler_cb client_handler, void* client_data);
```
The constructor creates a uniquely named **Custom** ODE Action. When invoked, this Action will call the Client provided callback function with the Frame Meta and Object Meta that triggered the ODE occurrence. 

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

### *dsl_ode_action_capture_area_new*
```C++
DslReturnType dsl_ode_action_capture_area_new(const wchar_t* name, const wchar_t* area, const wchar_t* outdir);
```
The constructor creates a uniquely named **Area Capture** ODE Action. When invoked, this Action will capture the rectangular area defined by and ODE Area object.  to a jpeg image file in the directory specified by `outdir`. The file name will be derived from combining the unique ODE Trigger name and unique ODE occurrence ID. The constructor will return `DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND` if `outdir` is invalid. 

Note: The Area is not required to owned by a Trigger.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `area` - [in] unique name for the ODE Area to capture.
* `outdir` - [in] absolute or relative path to the output directory to save the image file.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_capture_area_new('my-area-capture-action', 'my-area', './images/areas')
```

<br>

### *dsl_ode_action_capture_frame_new*
```C++
DslReturnType dsl_ode_action_capture_frame_new(const wchar_t* name, const wchar_t* outdir);
```
The constructor creates a uniquely named **Frame Capture** ODE Action. When invoked, this Action will capture the frame that triggered the ODE occurrence to a jpeg image file in the directory specified by `outdir`. The file name will be derived from combining the unique ODE Trigger name and unique ODE occurrence ID. The constructor will return `DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND` if `outdir` is invalid.

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
The constructor creates a uniquely named **Object Capture** ODE Action. When invoked, this Action will capture the object -- using its OSD rectangle parameters --  that triggered the ODE occurrence to a jpeg image file in the directory specified by `outdir`. The file name will be derived from combining the unique ODE Trigger name and unique ODE occurrence ID. The constructor will return `DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND` if `outdir` is invalid.

Note: Adding an Object Capture ODE Action to an Absence or Summation Trigger is meaningless and will result in a Non-Action.

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
The constructor creates a uniquely named **Display Data** ODE Action. When invoked, this Action writes the ODE Trigger's name and occurrence count as metadata to the current Frame Meta for display by a downstream On-Screen-Display (OSD) component.


**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `offsetX` - [in] offset for the display text in the X direction.
* `offsetY` - [in] offset for the display text in the Y direction.
* `offsetY_with_classId` - [in] if true adds an additional Y offset based on the Class Id of the Trigger invoking the Action. This allows multiple Triggers with different Class Ids to share the same Display Action

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_display_new('my-display-event-data-action', 10, 30, True)
```

<br>

### *dsl_ode_action_display_meta_add_new*
```C++
DslReturnType dsl_ode_action_display_meta_add_new(const wchar_t* name, const wchar_t* display_type);
```
The constructor creates a uniquely named **Add Display Meta** ODE Action. When invoked, this Action will add the Display Type to the Frame Meta that triggered the ODE occurrence.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `display_type` - [in] unique name of the Display Type to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_display_meta_add_new('my-add-display-meta-action', 'my-circle')
```

<br>

### *dsl_ode_action_display_meta_add_many_new*
```C++
DslReturnType dsl_ode_action_display_meta_add_new(const wchar_t* name, const wchar_t** display_types);
```
The constructor creates a uniquely named **Add Many Display Meta** ODE Action. When invoked, this Action will add many Display Types to the Frame Meta that triggered the ODE occurrence.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `display_types` - [in] NULL terminated list of unique names of the Display Types to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_display_meta_add_new('my-add-display-meta-action', 
    ['my-circle', 'my-rectangle', 'my-source-name', None])
```

<br>
### *dsl_ode_action_dump_new*
```C++
DslReturnType dsl_ode_action_dump_new(const wchar_t* name, const wchar_t* pipeline, const wchar_t* filename);
```
The constructor creates a uniquely named **Dump Pipeline Graph** ODE Action. When invoked, this Action dumps a Pipeline's graph to dot file. The GStreamer Pipeline creates a topology graph on each change of state to ready, playing and paused if the debug environment variable GST_DEBUG_DUMP_DOT_DIR is set.

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

### *dsl_ode_action_fill_frame_new*
```C++
DslReturnType dsl_ode_action_fill_frame_new(const wchar_t* name
    double red, double green, double blue, double alpha);
```
The constructor creates a uniquely named **Fill Frame** ODE Action. When invoked, this Action will fill the Frame with a rectangle background color added to the Frame Meta that triggered the ODE occurrence.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `color` - [in] RGBA Display Type color to fill the frame with.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_fill_frame_new('my-fill-frame-action', 'opaque-red')
```

<br>

### *dsl_ode_action_fill_object_new*
```C++
DslReturnType dsl_ode_action_fill_object_new(const wchar_t* name
    double red, double green, double blue, double alpha);
```
The constructor creates a uniquely named **Fill Object** ODE Action. When invoked, this Action will fill the Object's rectangle background color by updated the Metadata that triggered the ODE occurrence.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `color` - [in] RGBA Color Display Type to fill the object backround with.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_fill_object_new('my-fill-object-action', 'opaque-red')
```

<br>


### *dsl_ode_action_handler_disable_new*
```C++
DslReturnType dsl_ode_action_handler_disable_new(const wchar_t* name, const wchar_t* handler);
```
The constructor creates a uniquely named **Disable Handler** ODE Action. When invoked, this Action will disable a named ODE Handler. The action will produce an error log message if the Handler does not exist at the time of invocation.


**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `text` - [in] unique name for the ODE Handler to disable

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_handler_disable_new('my-disable-handler-action', 'my-handler)
```

<br>

### *dsl_ode_action_hide_new*
```C++
DslReturnType dsl_ode_action_hide_new(const wchar_t* name, boolean text, boolean border);
```
The constructor creates a uniquely named **Hide Display Meta** ODE Action. When invoked, this Action will hide the OSD display text and/or rectangle border for the Object Meta that triggered the ODE occurrence.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `text` - [in] if true, the action hides the display text for the Object that trigger the ODE occurrence
* `border` - [in] if true, the action hides the rectangle border for the Object that triggered the ODE occurrence

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_hide_new('my-hide-action', True, True)
```

<br>

### *dsl_ode_action_log_new*
```C++
DslReturnType dsl_ode_action_log_new(const wchar_t* name);
```
The constructor creates a uniquely named **GStreamer Log** ODE Action. When invoked, this Action will log ODE information from the Frame and Object metadata to the GStreamer logging service at a level of INFO using the DSL token. The GST_DEBUG level must be set to 4 or higher for the information to be logged. The following console example sets the default GStreamer level to ERROR=1 and DSL to INFO=4.
```
$ export GST_DEBUG=1,DSL:4 
```

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `text` - [in] if true, the action hides the display text for Object/Frame that triggered the ODE occurrence
* `border` - [in] if true, the action hides the rectangle border for Object that triggered the ODE occurrence

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_log_new('my-gstreamer-log-action')
```


<br>


### *dsl_ode_action_pause_new*
```C++
DslReturnType dsl_ode_action_pause_new(const wchar_t* name, const wchar_t* pipeline);
```
The constructor creates a uniquely named **Pause Pipeline** ODE Action. When invoked, this Action will pause a named Pipeline. The action will produce an error log message if the Pipeline does not exist at the time of invocation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `pipeline` - [in] the unique name for the Pipeline to pause on ODE occurrence

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_pause_new('my-pause-action', 'my-pipeline')
```

<br>

### *dsl_ode_action_print_new*
```C++
DslReturnType dsl_ode_action_print_new(const wchar_t* name);
```
The constructor creates a uniquely named **Print** ODE Action. When invoked, this Action will print the Frame, Object and Trigger information that triggered the ODE occurrence to the console. The Print action can be very useful when setting-up/testing new ODE Triggers and Areas.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_print_new('my-print-action')
```

<br>

### *dsl_ode_action_redact_new*
```C++
DslReturnType dsl_ode_action_redact_new(const wchar_t* name);
```
The constructor creates a uniquely named **Redact Object** ODE Action. When invoked, this Action will update an Object's metadata by 
* filling in the the rectangle with a black background color with an alpha level of 1.0. 
* hiding both the display text and rectangle boarder

The action will Redact any detected object based on the GIE model and class Id in use... Face, License, etc,

Note: Adding a Redact ODE Action to an Absence or Summation Trigger is meaningless and will result in a Non-Action.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_redact_new('my-redact-action')
```

<br>

### *dsl_ode_action_sink_add_new*
```C++
DslReturnType dsl_ode_action_sink_add_new(const wchar_t* name, 
  const wchar_t* pipeline, const wchar_t* sink);
```
The constructor creates a uniquely named **Add Sink** ODE Action. When invoked, this Action will attempt to add a named Sink to a named Pipeline. The action will produce an error log message if either the Pipeline or Sink do not exist at the time of invocation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `pipeline` - [in] the unique name for the Pipeline to add the Sink to on ODE occurrence
* `sink` - [in] the unique name for the Sink to add to the Pipeline on ODE occurrence

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_sink_add_new('my-add-sink-action', 'my-pipeline', 'my-sink')
```
<br>

### *dsl_ode_action_sink_remove_new*
```C++
DslReturnType dsl_ode_action_sink_remove_new(const wchar_t* name, 
  const wchar_t* pipeline, const wchar_t* sink);
```
The constructor creates a uniquely named **Remove Sink** ODE Action. When invoked, this Action will remove a named Sink from a named Pipeline. The action will produce an error log message if either the Pipeline or Sink do not exist at the time of invocation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `pipeline` - [in] the unique name for the Pipeline to remove the Sink from on ODE occurrence
* `sink` - [in] the unique name for the Sink to remove from the Pipeline on ODE occurrence

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_sink_remove_new('my-remove-sink-action', 'my-pipeline', 'my-sink')
```

<br>

### *dsl_ode_action_sink_record_start_new*
```C++
DslReturnType dsl_ode_action_sink_record_start_new(const wchar_t* name,
    const wchar_t* record_sink, uint start, uint duration, void* client_data);
```
The constructor creates a uniquely named **Start Record Sink** ODE Action. When invoked, this Action will a Record session for the named Sink. The action will produce an error log message if the Sink does not exist at the time of invocation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `sink` - [in] the unique name of the Record Sink to start.
* `start` - [in] start time before current time in seconds, should be less the Record Sink's cache size.
* `duration` - [in] duration of the recording in seconds
* `client_data` - [in] opaque pointer to client data

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_sink_record_start_new('my-start-record-sink-action', 'my-record-sink', 15, 360, Null)
```

<br>

### *dsl_ode_action_source_add_new*
```C++
DslReturnType dsl_ode_action_source_add_new(const wchar_t* name, 
  const wchar_t* pipeline, const wchar_t* source);
```
The constructor creates a uniquely named **Add Source** ODE Action. When invoked, this Action will attempt to add a named Sink to a named Pipeline. The action will produce an error log message if either the Pipeline or Sink do not exist at the time of invocation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `pipeline` - [in] the unique name for the Pipeline to add the Source to on ODE occurrence
* `sink` - [in] the unique name for the Source to add to the Pipeline on ODE occurrence

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_source_add_new('my-add-source-action', 'my-pipeline', 'my-source')
```
<br>

### *dsl_ode_action_source_remove_new*
```C++
DslReturnType dsl_ode_action_source_remove_new(const wchar_t* name, 
  const wchar_t* pipeline, const wchar_t* source);
```
The constructor creates a uniquely named **Remove Source** ODE Action. When invoked, this Action will attempt to remove a named Source from a named Pipeline. The action will produce an error log message if either the Pipeline or Source do not exist at the time of invocation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `pipeline` - [in] the unique name for the Pipeline to remove the Sink from on ODE occurrence
* `source` - [in] the unique name for the Sink to remove from the Pipeline on ODE occurrence

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_source_remove_new('my-remove-source-action', 'my-pipeline', 'my-source')
```

<br>

### *dsl_ode_action_tap_record_start_new*
```C++
DslReturnType dsl_ode_action_tap_record_start_new(const wchar_t* name,
    const wchar_t* record_sink, uint start, uint duration, void* client_data);
```
The constructor creates a uniquely named **Start Record Tap** ODE Action. When invoked, this Action will a Record session for the named Tap. The action will produce an error log message if the Tap does not exist at the time of invocation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `record_tap` - [in] the unique name of the Record Sink to start.
* `start` - [in] start time before current time in seconds, should be less the Record Sink's cache size.
* `duration` - [in] duration of the recording in seconds
* `client_data` - [in] opaque pointer to client data

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_tap_record_start_new('my-start-record-tap-action', 'my-record-tap', 15, 360, Null)
```

<br>

### *dsl_ode_action_tiler_source_show_new*
```C++
DslReturnType dsl_ode_action_tiler_source_show_new(const wchar_t* name, 
    const wchar_t* tiler, uint timeout, boolean has_precedence);
```
The constructor creates a uniquely named **Tiler Show Source** ODE Action. When invoked, this Action will call the named [Tiler](/docs/api-tiler.md) to show just the source (see [dsl_tiler_source_show_set](/docs/api-tiler.md#dsl_tiler_source_show_set) identified in the Frame's metadata that Triggered the ODE occurrence. The action will produce an error log message if the Tiler does not exist at the time of invocation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `tiler` - [in] the unique name of the Tiler to call on.
* `timeout` - [in] time to show the single source before switching back to "all sources" i.e. the tiled view. 
* `has_precedence` - [in] if true, will inform the Tiler that this source has precedence, even if another single source is being shown.

Note: setting `has_precedence` to true when using the action with a Trigger that has the `source_id` filter set to `DSL_ODE_ANY_SOURCE` (default) can lead to excessive switching. 

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_tiler_source_show_new('my-show-source-action', 'my-tiler', 20, False)
```

<br>

### *dsl_ode_action_trigger_disable_new*
```C++
DslReturnType dsl_ode_action_trigger_disable_new(const wchar_t* name, const wchar_t* trigger);
```
The constructor creates a uniquely named **Disable Trigger** ODE Action. When invoked, this Action will attempt to disable a named ODE Trigger. The Action will produce an error log message if the Trigger to disable does not exist at the time of invocation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `trigger` - [in] unique name for the ODE Trigger to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_trigger_disable_new('my-disable-trigger-action', 'my-trigger')
```

<br>

### *dsl_ode_action_trigger_enable_new*
```C++
DslReturnType dsl_ode_action_trigger_enable_new(const wchar_t* name, const wchar_t* trigger);
```
The constructor creates a uniquely named **Enable Trigger** ODE Action. When invoked, this Action will attempt to enable a named ODE Trigger. The Action will produce an error log message if the Trigger to enable does not exist at the time of invocation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `trigger` - [in] unique name for the ODE Trigger to enable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_trigger_enable_new('my-enable-trigger-action', 'my-trigger')
```

<br>

### *dsl_ode_action_trigger_reset_new*
```C++
DslReturnType dsl_ode_action_trigger_reset_new(const wchar_t* name, const wchar_t* trigger);
```
The constructor creates a uniquely named **Reset Trigger** ODE Action. When invoked, this Action will attempt to reset a named ODE Trigger. The Action will produce an error log message if the Trigger to reset does not exist at the time of invocation.

**Parameters**
* `name` - [in] unique name for the ODE Action to create.
* `trigger` - [in] unique name for the ODE Trigger to enable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_action_trigger_reset_new('my-enable-trigger-action', 'my-trigger')
```

<br>

---

## Destructors
### *dsl_ode_action_delete*
```C++
DslReturnType dsl_ode_action_delete(const wchar_t* action);
```
This destructor deletes a single, uniquely named ODE Action. The destructor will fail if the Action is currently `in-use` by one or more ODE Triggers

**Parameters**
* `action` - [in] unique name for the ODE Action to delete

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_action_delete('my-action')
```

<br>

### *dsl_ode_action_delete_many*
```C++
DslReturnType dsl_action_delete_many(const wchar_t** actions);
```
This destructor deletes multiple uniquely named ODE Actions. Each name is checked for existence, with the function returning `DSL_RESULT_ACTION_NAME_NOT_FOUND` on first occurrence of failure. The destructor will fail if one of the Actions is currently `in-use` by one or more ODE Triggers

**Parameters**
* `actions` - [in] a NULL terminated array of uniquely named ODE Actions to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_action_delete_many(['my-action-a', 'my-action-b', 'my-action-c', None])
```

<br>

### *dsl_ode_action_delete_all*
```C++
DslReturnType dsl_ode_action_delete_all();
```
This destructor deletes all ODE Actions currently in memory. The destructor will fail if any one of the Actions is currently `in-use` by one or more ODE Triggers. 

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_action_delete_all()
```

<br>
---

## Methods
### *dsl_ode_action_enabled_get*
```c++
DslReturnType dsl_ode_action_enabled_get(const wchar_t* name, boolean* enabled);
```
This service returns the current enabled setting for the named ODE Action. Note: Actions are enabled by default at the time of construction.

**Parameters**
* `name` - [in] unique name of the ODE Action to query.
* `enabled` - [out] true if the ODE Action is currently enabled, false otherwise

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, enabled = dsl_ode_action_enabled_get('my-action')
```

<br>

### *dsl_ode_action_enabled_set*
```c++
DslReturnType dsl_ode_action_enabled_set(const wchar_t* name, boolean enabled);
```
This service sets the enabled setting for the named ODE Action. Note: Actions are enabled by default at the time of construction.

**Parameters**
* `name` - [in] unique name of the ODE Action to update.
* `enabled` - [in] set to true to enable the ODE Action, false otherwise

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_action_enabled_set('my-action', False)
```

<br>

### *dsl_ode_action_list_size*
```c++
uint dsl_ode_action_list_size();
```
This service returns the size of the ODE Action container, i.e. the number of Actions currently in memory. 

**Returns**
* The size of the ODE Action container

**Python Example**
```Python
size = dsl_ode_action_list_size()
```

<br>
---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Source](/docs/api-source.md)
* [Tap](/docs/api-tap.md)
* [Dewarper](/docs/api-dewarper.md)
* [Primary and Secondary GIE](/docs/api-gie.md)
* [Tracker](/docs/api-tracker.md)
* [Tiler](/docs/api-tiler.md)
* [On-Screen Display](/docs/api-osd.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* **ODE Action**
* [ODE Area](/docs/api-ode-area.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
