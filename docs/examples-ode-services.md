# Object Detection Event (ODE) Services
This page covers examples using [ODE Pad Probe Handlers](/docs/api-pph.md#object-detection-event-ode-pad-probe-handler), [ODE Triggers](/docs/api-ode-trigger.md), [ODE Actions](/docs/api-ode-action.md), [ODE Areas](/docs/api-ode-area.md), [ODE Acummulators](/docs/api-ode-accumulator.md), and [ODE Heat Mappers](/docs/api-ode-heat-mapper.md).
* [Occurrence Trigger with Monitor Action](#occurrence-trigger-with-monitor-action)
* [Always Trigger with Display Metadata Action and Source Display Types](#always-trigger-with-display-metadata-action-and-source-display-types)
* [Minimum, Maximum, Range, and Summation Triggers, with Display Types and Actions](#minimum-maximum-range-and-summation-triggers-with-display-types-and-actions)
* [Distance Trigger with Format BBox Action](#distance-trigger-with-format-bbox-action)
* [New Instance Triggers and Print Action](#new-instance-triggers-and-print-action)
* [Intersection Triggers with Format BBox and Print Actions](#intersection-triggers-with-format-bbox-and-print-actions)
* [Largest Trigger with Fill Surroundings Action](#largest-trigger-with-fill-surroundings-action)
* [Cross Trigger with Line Area and ODE Accumulator with Display Action](#cross-trigger-with-line-area-and-ode-accumulator-with-display-action)
* [New High and New Low Count Triggers with Fill Frame and Print Event Data Actions](#new-high-and-new-low-count-triggers-with-fill-frame-and-print-event-data-actions)
* [Occurrence Trigger with Area of Inclusion or Exclusion](#occurrence-trigger-with-area-of-inclusion-or-exclusion)
* [Instance Trigger with Capture Frame Action with a Mailer to Email Frame as Attachement](#instance-trigger-with-capture-frame-action-with-a-mailer-to-email-frame-as-attachement)
* [Occurrence Trigger with ODE Head Mapper using RGBA Color Palettes](#occurrence-trigger-with-ode-head-mapper-using-rgba-color-palettes)
* [Persistence Triggers with Format BBox Actions](#persistence-triggers-with-format-bbox-actions)
* [Persistence and Earliest Triggers with Customize Label and Display Actions](#persistence-and-earliest-triggers-with-customize-label-and-display-actions)

<br> 

---

### Occurrence Trigger with Monitor Action 

* [`ode_occurrence_trigger_with_monitor_action.py`](/examples/python/ode_occurrence_trigger_with_monitor_action.py)
* cpp example is still to be done

```python
# 
# This example demonstrates the use of an "ODE Monitor Action" -- added to an 
# ODE Occurrence Trigger with the below criteria -- to monitor all 
# ODE Occurrences
#   - class id            = PGIE_CLASS_ID_VEHICLE
#   - inference-done-only = TRUE
#   - minimum confidience = VEHICLE_MIN_CONFIDENCE
#   - minimum width       = VEHICLE_MIN_WIDTH
#   - minimum height      = VEHICLE_MIN_HEIGHT
#
# The ode_occurrence_monitor callback function (defined below) is added to the 
# "Monitor Action" to be called with the ODE Occurrence event data for
# each detected object that meets the above criteria.
#  
# The application can process the event data as needed. This examples simply
# prints all of the event data to console.
#  
# The example uses a basic inference Pipeline consisting of:
#   - A URI Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - Two Secondary GST Inference Engines (SGIEs)
#   - On-Screen Display
#   - Window Sink
# 
```
<br> 

---

### Always Trigger with Display Metadata Action and Source Display Types 

* [`ode_always_trigger_display_source_info.py`](/examples/python/ode_always_trigger_display_source_info.py)
* cpp example is still to be done

```python
#
# This Example demonstrates how to use an ODE Always Trigger to update the 
# metadata of every frame to display specific information for each Source.
#
# 4 Sources are used, each with unique camara names.
#  
# 3 Display Types are used to create the metadata to be added to each frame:
#   * Source Stream Id
#   * Source Name
#   * Source Dimensions
#
# The 3 Display Types are added to an "Add Display Meta Action" which
# adds the metadata to a given frame.
#
# The ODE Action is added to an "Always Trigger" that always triggers once
# per frame in every batched frame (requires source=DSL_ODE_ANY_SOURCE).
#
# The ODE Trigger is added to a "ODE Pad Probe Handler" that is added
# to the sink (input) pad of the 2D Tiler. The ODE Handler is called with
# every batched frame that crosses over the Tilers sink pad.
# 
# The example uses a basic inference Pipeline consisting of:
#   - 4 URI Sources
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - 2D Tiler
#   - On-Screen Display
#   - Window Sink
#  
```

![](/Images/ode_always_trigger_display_source_info.png)
<br>

---

### Minimum, Maximum, Range, and Summation Triggers, with Display Types and Actions 

* [`ode_always_trigger_display_source_info.py`](/examples/python/ode_always_trigger_display_source_info.py)
* cpp example is still to be done

```python
#
# This example demonstrates how to use Minimum, Maximum, and Range Triggers.
#
# The triggers, upon meeting all criteria, will add a small rectangle (using 
# a "Display Type" and "Add Display Metadata Action") on the Frame with the 
# following colors indicating: 
#    Yellow = object count below Minimum
#    Red = object count above Maximum 
#    Green = object count in range of Minimim to Maximum.
#
# An additional "Summation Trigger" with a "Display Action" will display the 
# total number of objects next to the colored/filled indicator (rectangle)
#  
# The ODE Triggers are added to an ODE Pad Probe Handler which is added to
# source (output) pad of the Tracker. 

# The example uses a basic inference Pipeline consisting of:
#   - A URI Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
#  
```

![](/Images/ode_count_trigger_display_meta.png)
<br>

---

### Distance Trigger with Format BBox Action  

* [`ode_distance_trigger_fill_object.py`](/examples/python/ode_distance_trigger_fill_object.py)
* cpp example is still to be done

```python
#
# This example demonstrates the use of an ODE Distance Trigger to trigger on the
# occurrence of two objects of different class id that are closer that a 
# minimum distance - specifically testing the distance between People and Vehicles.
# The bounding boxes for the two objects that are witin the minimim distance will. 
# be filled (using a Format BBox Action) with a color for visual indication of 
# the events.
        
# The Distance trigger is created with minimim distance critera as a percentage
# of the width of Class A in the A/B distance measurement. In this example,
# Class A will be the Person class and Class B the Vehicle class. ODE Occurrence 
# will be triggered if the distance between any Person and Vehicle is measured to 
# be less 250% of the width of the Person's BBox. Maximum is set to 0 == no maximum.
# Note: Class A and Class B can be set to the same Class Id or DSL_ODE_ANY_CLASS.
# test_point is DSL_BBOX_POINT_SOUTH == measuring from center points of bottom edges.
# test_method is DSL_DISTANCE_METHOD_PERCENT_WIDTH_A == % of Person's BBox width.
#  
# The example uses a basic inference Pipeline consisting of:
#   - A URI Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
#  
```

![](/Images/ode_distance_trigger_fill_object.png)

<br>

---

### New Instance Triggers and Print Action

* [`ode_instance_trigger_print_action.py`](/examples/python/ode_instance_trigger_print_action.py)
* cpp example is still to be done


```python
#
# This example demonstrates the use of two ODE Instance Triggers -- one for 
# the Person class and the other for the Vehicle class -- to trigger on new 
# Object instances as identified by an IOU Tracker. A Print Action is added 
# to the Instance Triggers to print out the event data for each new object. 
#  
# The example uses a basic inference Pipeline consisting of:
#   - A URI Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
#  
```
<br>

An example of the Object Dection Event (ODE) data printed to the console by the ODE Print Trigger

```
Trigger Name        : person-instance-trigger
  Unique ODE Id     : 249
  NTP Timestamp     : 2024-09-14 14:13:10.084472
  Source Data       : ------------------------
    Inference       : Yes
    Source Id       : 0x00000000
    Batch Id        : 0
    Pad Index       : 0
    Frame           : 18
    Width           : 1920
    Heigh           : 1080
  Object Data       : ------------------------
    Obj ClassId     : 2
    Infer Id        : 1
    Tracking Id     : 1
    Label           : person
    Infer Conf      : 0.260092
    Track Conf      : 1
    Persistence     : 0
    Direction       : 0
    Left            : 405
    Top             : 464
    Width           : 65
    Height          : 240
  Criteria          : ------------------------
    Class Id        : 2
    Infer Id        : -1
    Min Infer Conf  : 0
    Min Track Conf  : 0
    Min Frame Count : 1 out of 1
    Min Width       : 0
    Min Height      : 0
    Max Width       : 0
    Max Height      : 0
    Inference       : No
```
<br>

---

### Intersection Triggers with Format BBox and Print Actions

* [`ode_intersection_trigger_min_max_dimensions.py`](/examples/python/ode_intersection_trigger_min_max_dimensions.py)
* cpp example is still to be done

```python
#
# This example is used to demonstrate the Use of Two Intersection Triggers, 
# one for the Vehicle class the other for the Person class. A "Format BBox" 
# action will be used to shade the background of the Objects intersecting.  
# Person intersecting with Person and Vehicle intersecting with Vehicle.
# 
# Min and Max Dimensions will set as addional criteria for the Preson and 
# Vehicle Triggers respecively
#  
# The example uses a basic inference Pipeline consisting of:
#   - A URI Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
#  
```
![](/Images/ode_intersection_trigger_min_max_dimensions.png)

<br>

---

### Largest Trigger with Fill Surroundings Action

* [`ode_largest_object_fill_surroundings.py`](/examples/python/ode_largest_object_fill_surroundings.py)
* cpp example is still to be done

```python
#
# This example demonstrates the used of a "Largest Object Trigger" and "Fill 
# Surroundings Action" to continuosly highlight the largest object in the Frame 
# as measured by bounding box area.
#
# The example uses a basic inference Pipeline consisting of:
#   - A URI Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
#  
```
![](/Images/ode_largest_object_fill_surroundings.png)


<br>

---

### Cross Trigger with Line Area and ODE Accumulator with Display Action  

* [`ode_line_cross_object_capture_overlay_image.py`](/examples/python/ode_line_cross_object_capture_overlay_image.py)
* cpp example is still to be done

```python
#
# This example demonstrates the use of an ODE Cross Trigger with an ODE Line Area 
# and ODE Accumulator to accumulate occurrences of an object (person) crossing 
# the line. The Accumulator uses an ODE Display Action to add the current counts 
# of the IN and OUT crossings as display-metadata to each frame.
#
# The bounding box and historical trace of each object (tracked by the "Cross 
# Trigger") is assigned a new random RGBA color and added as display-metadata 
# to each frame.
#
# An ODE Capture Object Action with an Image Render Player is added to the Cross
# Trigger to capture and render an image of each object (person) that crosses the 
# line. Each image is displayed for 3 seconds. All files are written to the current
# directory (configurable).
#
# The example uses a basic inference Pipeline consisting of:
#   - A File Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
#  
```
![](/Images/line-cross-capture-overlay-object-image.png)

<br>

---

### New High and New Low Count Triggers with Fill Frame and Print Event Data Actions

* [`ode_new_high_new_low_triggers_fill_frame.py`](/examples/python/ode_new_high_new_low_triggers_fill_frame.py)
* cpp example is still to be done

```python
#
# This example demonstrates the use of the New-High and New-Low Count Triggers
# that trigger on new high and low object counts respectively. The frame is 
# filled with a full color for a (brief) visual indication on each new occurrence.
# A print Action is used to print the event data to the console as well.
#
# The example uses a basic inference Pipeline consisting of:
#   - A URI Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
#  
```

<br>

---

### Occurrence Trigger with Area of Inclusion or Exclusion

* [`ode_occurrence_polygon_area_inclussion_exclusion.py`](/examples/python/ode_occurrence_polygon_area_inclussion_exclusion.py)
* cpp example is still to be done

```python
#
# This example demonstrates the use of a Polygon Area for Inclusion
# or Exclusion criteria for ODE occurrence.
#
# A "Polygon Display Type" is used to create either an ODE Area of Inclusion or
# Exclusion based on the AREA_TYPE variable defined below.
#
# The ODE Area is then added to an ODE Occurrence Trigger to be used as criteria
# for ODE occurrence.
#
# A "Format BBox Action" is used to fill each detected object that triggers
# occurrence with an opaque red color for visual confirmation.
#
# The example uses a basic inference Pipeline consisting of:
#   - A URI Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
# 
```
![](/Images/ode_occurrence_polygon_area_inclussion_exclusion.png)

<br>

---

### Instance Trigger with Capture Frame Action with a Mailer to Email Frame as Attachement

* [`ode_instance_frame_capture_email_attachment.py`](/examples/python/ode_instance_frame_capture_email_attachment.py)
* cpp example is still to be done

```python
#
# This example demostrates the use of an "ODE Occurrence Trigger" to trigger
# on every occurrence of every Person within a Polygon ODE Inclusion Area.
# The Trigger uses a "Format BBox Action" to fill each occurrence with
# an opaque red color for visual confirmation while the Person is in the Area.
#
# An Instance Trigger is then used to Trigger on every new Instance detected in
# the same ODE Area.. i.e. when the Person is first detected in the Area and only
# once.
# This Trigger uses a "Frame Capture Action" to capture and encode the frame
# and save it to file. The Action then uses a Mailer component to mail the
# image as an attachment using DSL's SMTP services.
#
# IMPORTANT! it is STRONGLY advised that you create a new, free Gmail account -- 
# that is seperate/unlinked from all your other email accounts -- strictly for 
# the purpose of sending ODE Event data uploaded from DSL.  Then, add your 
# Personal email address as a "To" address to receive the emails.
#
# Gmail considers regular email programs (i.e Outlook, etc.) and non-registered 
# third-party apps to be "less secure". The email account used for sending email 
# must have the "Allow less secure apps" option turned on. Once you've created 
# this new account, you can go to the account settings and enable Less secure 
# app access. see https://myaccount.google.com/lesssecureapps
#
# The example uses a basic inference Pipeline consisting of:
#   - A URI Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
#  
```

<br>

---

### Occurrence Trigger with ODE Head Mapper using RGBA Color Palettes

* [`ode_occurrence_trigger_with_heat_mapper.py`](/examples/python/ode_occurrence_trigger_with_heat_mapper.py)
* cpp example is still to be done

```python
#
# This example demonstrates the use of an ODE Heat-Mapper added to an 
# ODE Occurrence trigger that triggers on every Person occurrence.
# The occurrence data is mapped/ovelaid on everyframe. The example creates 
# all 5 predefined RGBA Color Palettes - Spectral, Red, Green, Blue, and Grey.
# The ODE Heat-Mapper is created with the Spectral palette, but can be updated
# at runtime by pressing the 'N' key.
#
# Several keys, bound to the Window Sink, are mapped to the ODE Heat Mapper services  
#    - 'N' key maps to 'next' color palette with - dsl_ode_heat_mapper_color_palette_set
#    - 'C' key maps to 'clear' heat-map metrics  - dsl_ode_heat_mapper_metrics_clear
#    - 'P' key maps to 'print' heat-map metrics  - dsl_ode_heat_mapper_metrics_print
#    - 'L' key maps to 'log' heat-map metrics    - dsl_ode_heat_mapper_metrics_log
#    - 'G' key maps to 'get' heat-map metrics    - dsl_ode_heat_mapper_metrics_get
#
#
# The example uses a basic inference Pipeline consisting of:
#   - A URI Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
# 
```
![](/Images/spectral-person-heat-map.png)

<br>

---

### Persistence Triggers with Format BBox Actions

* [`ode_persistence_trigger_fill_tracked_objects.py`](/examples/python/ode_persistence_trigger_fill_tracked_objects.py)
* cpp example is still to be done

```python
#
# This example demonstrates the use of three ODE Persistence Triggers to trigger on
# all tracked Objects - as identified by an IOU Tracker - that persist accross consecutive
# frames for a specifid period of time. Each trigger specifies a range of minimum and
# maximum times of persistence. 
#   Trigger 1: 0 - 3 seconds - action = fill object with opaque green color
#   Trigger 2: 3 - 6 seconds - action = fill object with opaque yellow color
#   Trigger 3: 6 - 0 seconds - action = fill object with opaque red color
# This will have the effect of coloring an object by its time in view
#  
# The example uses a basic inference Pipeline consisting of:
#   - A URI Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
# 
```

<br>

---

### Persistence and Earliest Triggers with Customize Label and Display Actions

* [`ode_persistence_and_earliest_triggers_custom_labels.py`](/examples/python/ode_persistence_and_earliest_triggers_custom_labels.py)
* cpp example is still to be done

```python
#
# This script demonstrates the use of a Persistence Trigger to trigger on each Vehicle
# that is tracked for more than one frame -- to calculate the time of Object persistence
# from the first frame the object was detected.
#
# The Tracked Object's label is then "customized" to show the tracking Id and time of
# persistence for each tracked Vehicle.
#
# The script also creates an Earliest Trigger to trigger on the Vehicle that appeared
# the earliest -- i.e. the object with greatest persistence value -- and displays that
# Object's persistence using an ODE Display Action.
#
# The example uses a basic inference Pipeline consisting of:
#   - A URI Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
#
```
![](/Images/display-action-screenshot.png)
