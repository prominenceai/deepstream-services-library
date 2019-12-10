# Tiled Display API

## Tiled Display API
* [dsl_display_new](#dsl_display_new)
* [dsl_display_dimensions_get](#dsl_display_dimensions_get)
* [dsl_display_dimensions_set](#dsl_display_dimensions_set)
* [dsl_display_tiles_get](#dsl_display_tiles_get)
* [dsl_display_tiles_set](dsl_display_tiles_set)

## Return Values
The following return codes are used by the Pipeline API
```C++
#define DSL_RESULT_DISPLAY_NAME_NOT_UNIQUE                          0x10000001
#define DSL_RESULT_DISPLAY_NAME_NOT_FOUND                           0x10000010
#define DSL_RESULT_DISPLAY_NAME_BAD_FORMAT                          0x10000011
#define DSL_RESULT_DISPLAY_THREW_EXCEPTION                          0x10000100
#define DSL_RESULT_DISPLAY_IS_IN_USE                                0x10000101
#define DSL_RESULT_DISPLAY_SET_FAILED                               0x10000110
```

## Constructors
### *dsl_display_new*
```C++
DslReturnType dsl_display_new(const wchar_t* name, uint width, uint height);
```
The constructor creates a uniquely named Display. Construction will fail
if the name is currently in use.

**Parameters**
* `name` - unique name for the Tiled Display to create.
* `width` - width of the Tilded Display in pixels
* `width` - height of the Tilded Display in pixels

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

<br>

## Methods
### *dsl_display_dimensions_get*
```C++
DslReturnType dsl_display_dimensions_get(const wchar_t* name, uint* width, uint* height);
```
**Parameters**
* `name` - unique name for the Tiled Display to query.
* `width` - width of the Tiled Display in pixels.
* `width` - height of the Tiled Display in pixels.

**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_display_dimensions_set*
```C++
DslReturnType dsl_display_dimensions_set(const wchar_t* name, uint width, uint height);
```
**Parameters**
* `name` - unique name for the Tiled Display to update.
* `width` - current width of the Tiled Display in pixels.
* `width` - current height of the Tiled Display in pixels.

**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_display_tiles_get*
```C++
DslReturnType dsl_display_tiles_get(const wchar_t* name, uint* cols, uint* rows);
```
**Parameters**
* `name` - unique name for the Tiled Display to update.
* `cols` - current columns setting for the Tiled Display.
* `rows` - current rows setting for the Tiled Display.

**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_display_tiles_set*
```C++
DslReturnType dsl_display_tiles_set(const wchar_t* name, uint cols, uint rows);
```
**Parameters**
* `name` - unique name for the Tiled Display to update.
* `cols` - new columns setting for the Tiled Display.
* `rows` - new rows setting for the Tiled Display.

**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

<br>
