# ODE Area Services API

#### ODE Area Construction and Destruction

#### Adding/Removing ODE Areas

## ODE Area Services API

**Constructors:**
* [dsl_ode_area_new](#dsl_ode_area_new)

**Destructors:**
* [dsl_ode_area_delete](#dsl_ode_area_delete)
* [dsl_ode_area_delete_many](#dsl_ode_area_delete_many)
* [dsl_ode_area_delete_all](#dsl_ode_area_delete_all)

**Methods:**
* [dsl_ode_area_get](#dsl_ode_area_get)
* [dsl_ode_area_set](#dsl_ode_area_get)
* [dsl_ode_area_color_get](#dsl_ode_area_color_get)
* [dsl_ode_area_color_set](#dsl_ode_area_color_set)
* [dsl_ode_area_list_size](#dsl_ode_area_list_size)
---

## Return Values
The following return codes are used by the OSD Area API
```C++
#define DSL_RESULT_ODE_AREA_RESULT                                  0x00100000
#define DSL_RESULT_ODE_AREA_NAME_NOT_UNIQUE                         0x00100001
#define DSL_RESULT_ODE_AREA_NAME_NOT_FOUND                          0x00100002
#define DSL_RESULT_ODE_AREA_THREW_EXCEPTION                         0x00100003
#define DSL_RESULT_ODE_AREA_IN_USE                                  0x00100004
#define DSL_RESULT_ODE_AREA_SET_FAILED                              0x00100005
```

<br>

---
## Constructors
### *dsl_ode_area_new*
```C++
DslReturnType dsl_ode_area_new(const wchar_t* name, 
    uint left, uint top, uint width, uint height, boolean display);
```
The constructor creates a uniquely named ODE Area with coordinates and dimensions. The Area can be displayed (requires an On-Screen Display) or left hidden. Areas are created with a default background color of white, with an alpha level of 0.25. The background color can be changed by calling [dsl_ode_area_color_set](#dsl_ode_area_color_set)

**Parameters**
* `name` - [in] unique name for the ODE Area to create.
* `left` - [in] left coordinate for Area rectangle in pixels.
* `top` - [in] top coordinate for Area rectangle in pixels.
* `wdith` - [in] width for the Area rectangle in pixels.
* `height` - [in] height for the Area rectangle in pixels.
* `display` - [in] if true, rectangle display-metadata will be added to each structure of frame metadata.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_area_new('my-area', 120, 30, 700, 400, True)
```

<br>

---

## Destructors
### *dsl_ode_area_delete*
```C++
DslReturnType dsl_ode_area_delete(const wchar_t* area);
```
This destructor deletes a single, uniquely named ODE Area. The destructor will fail if the Area is currently `in-use` by one or more ODE Triggers

**Parameters**
* `area` - [in] unique name for the ODE Area to delete

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_area_delete('my-area')
```

<br>

### *dsl_ode_area_delete_many*
```C++
DslReturnType dsl_area_delete_many(const wchar_t** area);
```
This destructor deletes multiple uniquely named ODE Areas. Each name is checked for existence, with the function returning `DSL_RESULT_AREA_NAME_NOT_FOUND` on first occurrence of failure. The destructor will fail if one of the Areas is currently `in-use` by one or more ODE Triggers

**Parameters**
* `areas` - [in] a NULL terminated array of uniquely named ODE Areas to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_area_delete_many(['my-area-a', 'my-area-b', 'my-area-c', None])
```

<br>

### *dsl_ode_area_delete_all*
```C++
DslReturnType dsl_ode_area_delete_all();
```
This destructor deletes all ODE Areas currently in memory. The destructor will fail if any one of the Areas is currently `in-use` by one or more ODE Triggers. 

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_area_delete_all()
```

<br>

## Methods
### *dsl_ode_area_get*
```c++
DslReturnType dsl_ode_area_get(const wchar_t* name, 
    uint* left, uint* top, uint* width, uint* height, boolean *display);
```
This service returns a named ODE Area's current rectangle coordinates, dimensions, and display setting.

**Parameters**
* `name` - [in] unique name of the ODE Area to query.
* `left` - [out] left coordinate for Area rectangle in pixels.
* `top` - [out] top coordinate for Area rectangle in pixels.
* `wdith` - [out] width for the Area rectangle in pixels.
* `height` - [out] height for the Area rectangle in pixels.
* `display` - [out] if true, rectangle display-metadata will be added to each structure of frame metadata.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, left, top, width, height, display = dsl_ode_area_get('my-area')
```

<br>

### *dsl_ode_area_set*
```c++
DslReturnType dsl_ode_area_set(const wchar_t* name, 
    uint left, uint top, uint width, uint height, boolean display);
```
This service returns a named ODE Area's current rectangle coordinates, dimensions, and display setting.

**Parameters**
* `name` - [in] unique name of the ODE Area to update.
* `left` - [in] left coordinate param for Area rectangle in pixels.
* `top` - [in] top coordinate param for Area rectangle in pixels.
* `wdith` - [in] width param for the Area rectangle in pixels.
* `height` - [in] height param for the Area rectangle in pixels.
* `display` - [in] if true, rectangle display-metadata will be added to each structure of frame metadata.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_area_set('my-area', 0, 0. 128, 1028, False)
```

### *dsl_ode_area_color_get*
```c++
DslReturnType dsl_ode_area_color_get(const wchar_t* name, 
    double* red, double* green, double* blue, double* alpha);
```
This service returns a named ODE Area's current RGBA background color.

**Parameters**
* `name` - [in] unique name of the ODE Area to query.
* `red` - [out] red color value for the Area's RGBA background color [0.0..1.0].
* `green` - [out] green color value for the Area's RGBA background color [0.0..1.0].
* `blue` - [out] blue color value for the Area's RGBA background color [0.0..1.0].
* `alpha` - [out] alpha color value for the Area's RGBA background color [0.0..1.0].

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, left, top, width, height, display = dsl_ode_area_get('my-area')
```

<br>

### *dsl_ode_area_color_set*
```c++
DslReturnType dsl_ode_area_set(const wchar_t* name, 
    uint left, uint top, uint width, uint height, boolean display);
```
This service returns a named ODE Area's current rectangle coordinates, dimensions, and display setting.

**Parameters**
* `name` - [in] unique name of the ODE Area to update.
* `red` - [in] red color value for the Area's RGBA background color [0.0..1.0].
* `green` - [in] green color value for the Area's RGBA background color [0.0..1.0].
* `blue` - [in] blue color value for the Area's RGBA background color [0.0..1.0].
* `alpha` - [in] alpha color value for the Area's RGBA background color [0.0..1.0].

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_area_set('my-area', 0, 0. 128, 1028, False)
```

<br>
### *dsl_ode_area_list_size*
```c++
uint dsl_ode_area_list_size();
```
This service returns the size of the ODE Area container, i.e. the number of Areas currently in memory. 

**Returns**
* The size of the ODE Area container

**Python Example**
```Python
size = dsl_ode_area_list_size()
```

<br>
---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Source](/docs/api-source.md)
* [Dewarper](/docs/api-dewarper.md)
* [Primary and Secondary GIE](/docs/api-gie.md)
* [Tracker](/docs/api-tracker.md)
* [Tiler](/docs/api-tiler.md)
* [ODE Action](/docs/api-ode-action.md)
* **ODE-Area**
* [On-Screen Display](/docs/api-osd.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [Sink](/docs/api-sink.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
