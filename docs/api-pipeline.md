# Pipeline API Refernce
Pipelines are the top level Component in DSL. Pipelines manage and synchronize their Child componets as you set them to `playing`, `paused`, or `stopped`. 

Pipelines are constructed by calling [dsl_pipeline_new](#dsl_pipeline_new) or [dsl_pipeline_new_many](#dsl_pipeline_new_many). Information about the current Pipelines in memory can be obtained by calling [dsl_pipeline_list_size](#dsl_pipeline_list_size) and 
[dsl_pipeline_list_all](#dsl_pipeline_list_all).

Child components - such as Streaming Sources, Infer-engines, Displays, and Sinks - are added to a Pipeline by calling [dsl_pipeline_component_add](#dsl_pipeline_component_add) and [dsl_pipeline_component_add_many](#dsl_pipeline_component_add_many). Information about a Pipeline's current Child components of a Pipeline can be obtained by calling [dsl_pipeline_component_list_size](#dsl_pipeline_component_list_size) and [dsl_pipeline_component_list_all](#dsl_pipeline_component_list_all)

Pipelines - with a minimum required set of components - can be played by calling [dsl_pipeline_play](#dsl_pipeline_play), puased by calling [dsl_pipeline_pause](#dsl_pipeline_pause) and [dsl_pipeline_stop](#dsl_pipeline_stop).

Child components can be removed from their Parent Pipeline by calling [dsl_pipeline_component_remove](#dsl_pipeline_componet_remove), [dsl_pipeline_component_remove_many](#dsl_pipeline_componet_remove_many), and [dsl_pipeline_component_remove_all](#dsl_pipeline_component_remove_all)

Clients can be notified of Pipeline State-Changes by registering one or more callback functions with [dsl_pipeline_state_change_listener_add](#dsl_pipeline_state_change_listener_add). Notifications are stopped by calling [dsl_pipeline_state_change_listener_remove](#dsl_pipeline_state_change_listener_remove).

Pipelines that have at least one Overlay-Sink will create an XWindow by default. Clients can optain a handle to this window by calling [dsl_pipeline_display_window_handle_get](#dsl_pipeline_display_window_handle_get). Conversely, the Client can provide the Pipeline with the XWindow handle to use for it's XDisplay by calling [dsl_pipeline_display_window_handle_set](#dsl_pipeline_display_window_handle_set).

Clients can be notified of XWindow events - key & and button presses - with information by registering one or more callback funtions with [dsl_pipeline_display_event_handler_add](#dsl_pipeline_display_event_handler_add). Notifications are stopped by calling [dsl_pipeline_display_event_handler_remove](#dsl_pipeline_display_event_handler_remove).

Pipelines are destructed by calling [dsl_pipeline_delete](#dsl_pipeline_delete), [dsl_pipeline_delete_many](#dsl_pipeline_delete_many), or [dsl_pipeline_delete_all](#dsl_pipeline_delete_all). Deleting a pipeline will return all Child components to a state of `not_in_use`. It is up to calling application to delete all Child components by calling [dsl_component_delete](/docs/api-component.md#dsl_component_delete), [dsl_component_delete_many](/docs/api-component.md#dsl_component_delete_many), [dsl_component_delete_all](/docs/api-component.md#dsl_component_delete_all).


## Pipeline API
* [dsl_pipeline_new](#dsl_pipeline_new)
* [dsl_pipeline_new_many](#dsl_pipeline_new_many)
* [dsl_pipeline_delete](#dsl_pipeline_delete)
* [dsl_pipeline_delete_many](#dsl_pipeline_delete_many)
* [dsl_pipeline_delete_all](#dsl_pipeline_delete_all)
* [dsl_pipeline_list_size](#dsl_pipeline_list_size)
* [dsl_pipeline_list_all](#dsl_pipeline_list_all)
* [dsl_pipeline_component_add](#dsl_pipeline_component_add)
* [dsl_pipeline_component_add_many](#dsl_pipeline_component_add_many)
* [dsl_pipeline_component_list_size](#dsl_pipeline_component_list_size)
* [dsl_pipeline_component_list_all](#dsl_pipeline_component_list_all)
* [dsl_pipeline_component_remove](#dsl_pipeline_component_remove)
* [dsl_pipeline_component_remove_many](#dsl_pipeline_component_remove_many)
* [dsl_pipeline_component_remove_all](#dsl_pipeline_component_remove_all)
* [dsl_pipeline_component_replace](#dsl_pipeline_component_replace)
* [dsl_pipeline_streammux_properties_get](#dsl_pipeline_streammux_properties_get)
* [dsl_pipeline_streammux_properties_set](#dsl_pipeline_streammux_properties_set)
* [dsl_pipeline_play](#dsl_pipeline_play)
* [dsl_pipeline_pause](#dsl_pipeline_pause)
* [dsl_pipeline_stop](#dsl_pipeline_stop)
* [dsl_pipeline_state_get](#dsl_pipeline_state_get)
* [dsl_pipeline_state_change_listener_add](#dsl_pipeline_state_change_listener_add)
* [dsl_pipeline_state_change_listener_remove](#dsl_pipeline_state_change_listener_remove)
* [dsl_pipeline_display_window_handle_get](#dsl_pipeline_display_window_handle_get)
* [dsl_pipeline_display_window_handle_set](#dsl_pipeline_display_window_handle_set)
* [dsl_pipeline_display_event_handler_add](#dsl_pipeline_display_event_handler_add)
* [dsl_pipeline_display_event_handler_remove](#dsl_pipeline_display_event_handler_remove)
* [dsl_pipeline_dump_to_dot](#dsl_pipeline_dump_to_dot)
* [dsl_pipeline_dump_to_dot_with_ts](#dsl_pipeline_dump_to_dot_with_ts)

## Return Values
The following return codes are used by the Pipeline API
```C++
#define DSL_RESULT_PIPELINE_RESULT                                  0x11000000
#define DSL_RESULT_PIPELINE_NAME_NOT_UNIQUE                         0x11000001
#define DSL_RESULT_PIPELINE_NAME_NOT_FOUND                          0x11000010
#define DSL_RESULT_PIPELINE_NAME_BAD_FORMAT                         0x11000011
#define DSL_RESULT_PIPELINE_STATE_PAUSED                            0x11000100
#define DSL_RESULT_PIPELINE_STATE_RUNNING                           0x11000101
#define DSL_RESULT_PIPELINE_NEW_EXCEPTION                           0x11000110
#define DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED                    0x11000111
#define DSL_RESULT_PIPELINE_STREAMMUX_SETUP_FAILED                  0x11001000
#define DSL_RESULT_PIPELINE_FAILED_TO_PLAY                          0x11001001
#define DSL_RESULT_PIPELINE_FAILED_TO_PAUSE                         0x11001010
#define DSL_RESULT_PIPELINE_LISTENER_NOT_UNIQUE                     0x11001011
#define DSL_RESULT_PIPELINE_LISTENER_NOT_FOUND                      0x11001100
#define DSL_RESULT_PIPELINE_HANDLER_NOT_UNIQUE                      0x11001101
#define DSL_RESULT_PIPELINE_HANDLER_NOT_FOUND                       0x11001110
#define DSL_RESULT_PIPELINE_SUBSCRIBER_NOT_UNIQUE                   0x11010001
#define DSL_RESULT_PIPELINE_SUBSCRIBER_NOT_FOUND                    0x11010010
```

## Constructors
### *dsl_pipeline_new*
```C++
DslReturnType dsl_pipeline_new(const wchar_t* pipeline);
```
The constructor creates a uniquely named Pipeline. Construction will fail
if the name is currently in use.

**Parameters**
* `pipeline` - unique name for the Pipeline to create.

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_pipeline_new_many*
```C++
DslReturnType dsl_pipeline_new_many(const wchar_t** pipelines);
```
The constructor creates multiple uniquely named Pipelines at once. All names are checked for uniqueness before any contruction takes place. The call will fail if any of the names are duplicates, or are currently in use.

**Parameters**
* `pipelines` - a NULL terminated array of unique names for the Pipelines to create.

**Returns**
`DSL_RESULT_SUCCESS` on successful creation of all Pipelines. One of the [Return Values](#return-values) defined above on failure

<br>

## Destructors
### *dsl_pipeline_delete*
```C++
DslReturnType dsl_pipeline_delete(const wchar_t* pipeline);
```
This destructor deletes a single, uniquely named Pipeline. 
All components owned by the pipeline move to a state of `not-in-use`.

**Parameters**
* `pipelines` - unique name for the Pipeline to delete

**Returns**
`DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_pipeline_delete_many*
```C++
DslReturnType dsl_pipeline_delete_many(const wchar_t** pipelines);
```
This destructor deletes multiple uniquely named Pipelines. All names are first checked for existence. 
The function returns DSL_RESULT_PIPELINE_NAME_NOT_FOUND on first occurrence of not found, before making any deletions. 
All components owned by the Pipelines move to a state of `not-in-use`

**Parameters**
* `pipelines` - a NULL terminated array of uniquely named Pipelines to delete.

**Returns**
`DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_pipeline_delete_all*
```C++
DslReturnType dsl_pipeline_delete_all();
```
This destructor deletes all Pipelines currently in memory  All components owned by the pipelines move to a state of `not-in-use`

**Returns**
`DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

<br>

## Methods
### *dsl_pipeline_list_size*
```C++
uint dsl_pipeline_list_size();
```
This method returns the size of the current list of Pipelines in memory

**Returns** the number of Pipelines currently in memory

<br>

### *dsl_pipeline_list_all*
```C++
const wchar_t* dsl_pipeline_list_all();
```
This method returns the list of Pipelines currently in  memory

**Returns** a NULL terminated array of Pipeline names

<br>

### *dsl_pipeline_component_add*
```C++
DslReturnType dsl_pipeline_component_add(const wchar_t* pipeline, const wchar_t* component);
```
Adds a single named Component to a named Pipeline. The add service will fail if the component is currently `in-use` by any Pipeline. The add service will fail if adding a `one-only` type of Component, such as a Tiled-Display, for which the Pipeline already has. The Component's `in-use` state will be set to *true* on successful add. 

If a Pipeline is in a `playing` or `paused` state, the service will attempt a dynamic update if possible, returning from the call with the Pipeline in the same state.

**Parameters**
* `pipeline` - unique name for the Pipeline to update.
* `component` - unique name of the Component to add.

**Returns**
`DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_pipeline_component_add_many*
```C++
DslReturnType dsl_pipeline_component_add_many(const wchar_t* pipeline, const wchar_t** components);
```
Adds a list of named Component to a named Pipeline. The add service will fail if any of components are currently `in-use` by any Pipeline. The add service will fail if any of the components to add are a `one-only` type of component for which the Pipeline already has. All of the component's `in-use` states will be set to true on successful add. 

If a Pipeline is in a `playing` or `paused` state, the service will attempt a dynamic update if possible, returning from the call with the Pipeline in the same state.

**Parameters**
* `pipeline` - unique name for the Pipeline to update.
* `components` - a NULL terminated array of uniquely named Components to add.

**Returns**
`DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_pipeline_component_list_size*
```C++
uint dsl_pipeline_list_size(wchar_t* pipeline);
```
This method returns the size of the current list of Components `in-use` by the named Pipeline

**Parameters**
* `pipeline` - unique name for the Pipeline to query.

**Returns** the size of the list of Components currently in use

<br>

### *dsl_pipeline_component_list_all*
```C++
const wchar_t* dsl_pipeline_component_list_all(wchar_t* pipeline);
```
This method returns the list of Components currently `in-use` by the named Pipeline 

**Returns** a NULL terminated array of Component names

<br>

### *dsl_pipeline_component_remove*
```C++
DslReturnType dsl_pipeline_component_remove(const wchar_t* pipeline, const wchar_t* component);
```
Removes a single named Component from a named Pipeline. The remove service will fail if the Component is not currently `in-use` by the Pipeline. The Component's `in-use` state will be set to *false* on successful removal. 

If a Pipeline is in a `playing` or `paused` state, the service will attempt a dynamic update if possible, returning from the call with the Pipeline in the same state.

**Parameters**
* `pipeline` - unique name for the Pipeline to update.
* `component` - unique name of the Component to remove.

**Returns**
`DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_pipeline_component_remove_many*
```C++
DslReturnType dsl_pipeline_component_add_many(const wchar_t* pipeline, const wchar_t** components);
```
Removes a list of named components from a named Pipeline. The add service will fail if any of components are currently `not-in-use` by the named Pipeline.  All of the removed component's `in-use` state will be set to *false* on successful removel. 

If a Pipeline is in a `playing` or `paused` state, the service will attempt a dynamic update if possible, returning from the call with the Pipeline in the same state.

**Parameters**
* `pipeline` - unique name for the Pipeline to update.
* `components` - a NULL terminated array of uniquely named Components to add.

**Returns**
`DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_pipeline_component_replace*
```C++
DslReturnType dsl_pipeline_component_replace(const wchar_t* pipeline, const wchar_t* prev, const wchar_t* next);
```
Replaces a single Component `in-use` by the named Pipeline with a new Component of the same type. The replace service will fail under the if the two component are of different types, or if the new Component is already `in-use` by another Pipele. The previous Component's `in-use` state will be set to *false* and the new Component's state to *true* on successful replacement.  

**Parameters**
* `pipeline` - unique name for the Pipeline to update.
* `prev` - unique name of the Component to replace... becoming the previous.
* `next` - unique name of the Component to use next... in place of the previous.

**Returns**
`DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_pipeline_streammux_properties_get*

**TODO**

<br>

### *dsl_pipeline_streammux_properties_set*

**TODO**

<br>

### *dsl_pipeline_play*
```C++
bool dsl_pipeline_play(wchar_t* pipeline);
```
This service is used to play a named Pipeline. The service will fail if the Pipeline's list of components are insufficient for the Pipeline to play. The service will also fail if one of the Pipeline's components fails to transition to a state of `playing`.  

**Parameters**
* `pipeline` - unique name for the Pipeline to play.

**Returns** *true* if the named Pipeline is able to succesfully transition to a state of `playing`

<br>

### *dsl_pipeline_pause*
```C++
bool dsl_pipeline_pause(wchar_t* pipeline);
```
This service is used to pause a named Pipeline. The service will fail if the Pipeline is not currently in a `playing` state. The service will also fail if one of the Pipeline's components fails to transition to a state of `paused`.  

**Parameters**
* `pipeline` - unique name for the Pipeline to pause.

**Returns** *true* if the named Pipeline is able to succesfully transition to a state of `paused`, *false* otherwise.

<br>

### *dsl_pipeline_stop*
```C++
bool dsl_pipeline_stop(wchar_t* pipeline);
```
This service is used to stop a named Pipeline. The service will fail if the Pipeline is not currently in a `playing` or `paused` state. The service will also fail if one of the Pipeline's components fails to transition to a state of `stoped`.  

**Parameters**
* `pipeline` - unique name for the Pipeline to stop.

**Returns** *true* if the named Pipeline is able to succesfully transition to a state of `stopped`, *false* otherwise.

<br>

### *dsl_pipeline_state_get*
```C++
unit dsl_pipeline_state_get(wchar_t* pipeline);
```
This service returns the current [state]() of the named Pipeline The service fails if the named Pipeline was not found.  

**Parameters**
* `pipeline` - unique name for the Pipeline to query.

**Returns** the current [state]() of the named Pipeline if found. One of the [Return Values](#return-values) defined above on failure.

<br>

### *dsl_pipeline_state_change_listener_add*
```C++
DslReturnType dsl_pipeline_state_change_listener_add(const wchar_t* pipeline, 
    state_change_listener_cb listener, void* user_data);
```
This service adds a callback function of type [dsl_state_change_listener_cb](#dsl_state_change_listener_cb) to a
pipeline identified by it's unique name. The function will be called on every Pipeline `change-of-state` with 
current and previous state information and the client provided `user_data`. Multiple calback functions can be 
registered with one Pipeline, and one callback function can be registered with multiple Pipelines.

**Parameters**
* `pipeline` - unique name of the Pipeline to update.
* `listener` - state change listener callback function to add.
* `user_data` - opaque pointer to user data returned to the listner is called back

**Returns**  `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

<br>

### *dsl_pipeline_state_change_listener_remove*
```C++
DslReturnType dsl_pipeline_state_change_listener_remove(const wchar_t* pipeline, 
    state_change_listener_cb listener);
```
This service removes a callback function of type [state_change_listener_cb](#state_change_listener_cb) from a
pipeline identified by it's unique name.

<br>

### *dsl_pipeline_display_event_handler_add*
```C++
DslReturnType dsl_pipeline_display_event_handler_add(const wchar_t* pipeline, 
    dsl_display_event_handler_cb handler, void* user_data);
```
This service adds a callback function of type [dsl_display_event_handler_cb](#dsl_display_event_handler_cb) to a
pipeline identified by it's unique name. The function will be called on every Pipeline Window/Display [KeyPressed|ButtonPressed] with event info - `display_coordinates`, `source_name` and the client provided `user_data`. Multiple calback functions can be registered with one Pipeline, and one callback function can be registered with multiple Pipelines.

**Parameters**
* `pipeline` - unique name of the Pipeline to update.
* `handler` - window/display event handler callback function to add.
* `user_data` - opaque pointer to user data returned to the handler when called back

**Returns**  `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

<br>

### *dsl_pipeline_display_event_handler_remove*
```C++
DslReturnType dsl_pipeline_display_event_handler_remove(const char* pipeline, 
    dsl_display_event_handler_cb handler);
```

**Parameters**
* `pipeline` - unique name of the Pipeline to update
* `handler` - display event handler callback function to remove.

**Returns**  `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure.


### *dsl_pipeline_dump_to_dot*
```C++
DslReturnType dsl_pipeline_dump_to_dot(const char* pipeline, char* filename);
```
This method dumps a Pipeline's graph to dot file. The GStreamer Pipeline will a create 
topology graph on each change of state to ready, playing and paused if the debug 
enviornment variable `GST_DEBUG_DUMP_DOT_DIR` is set.

GStreamer will add the `.dot` suffix and write the file to the directory specified by
the environment variable. The caller of this service is responsible for providing a 
correctly formatted and filename. 

**Parameters**
* `pipeline` - unique name of the Pipeline to dump
* `filename` - name of the file without extension.

**Returns**  `DSL_RESULT_SUCCESS` on successful file dump. One of the [Return Values](#return-values) defined above on failure.

<br>

### *dsl_pipeline_dump_to_dot_with_ts*
```C++
DslReturnType dsl_pipeline_dump_to_dot_with_ts(const char* pipeline, char* filename);
```
This method dumps a Pipeline's graph to dot file prefixed with the current timestamp. 
Except for the prefix, this method performs the identical service as 
[dsl_pipeline_dump_to_dot](#dsl_pipeline_dump_to_dot).

<br>
