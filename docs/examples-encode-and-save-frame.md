# Different Methods of Encoding and Saving a Frame to JPEG File

<br>

---

* [`encode_and_save_frame_to_jpeg_from_custom_pph.py`](/examples/python/encode_and_save_frame_to_jpeg_from_custom_pph.py)
* [`encode_and_save_frame_to_jpeg_from_custom_pph.cpp`](/examples/cpp/encode_and_save_frame_to_jpeg_from_custom_pph.cpp)

```python
#
# This example demonstrates the use of a Frame-Capture Sink to encode and
# save video frames to JPEG files scheduled from a Custom Pad Probe Handler (PPH).
#
# An ODE Frame-Capture Action is provided to The Frame-Capture Sink on creation.
# A client "capture_complete_listener" is added to the the Action to be notified
# when each new file is saved (the ODE Action performs the actual frame-capture).
#
# Child Players (to play the captured image) and Mailers (to mail the image) can
# be added to the ODE Frame-Capture action as well (not shown).
#
# A Custom Pad Probe Handler (PPH) is added to the sink-pad of the OSD component
# to process every buffer flowing over the pad by:
#    - Retrieving the batch-metadata and its list of frame metadata structures
#      (only one frame per batched-buffer with 1 Source)
#    - Retrieving the list of object metadata structures from the frame metadata.
#    - Iterating through the list of objects looking for the first occurrence of
#      a bicycle. 
#    - If detected, the current frame-number is schedule to be captured by the
#      Frame-Capture Sink using its Frame-Capture Action.
#
#          dsl_sink_frame_capture_schedule('frame-capture-sink', 
#                   frame_meta.frame_num)
#
# Note: The Custom PPH will schedule every frame with a bicycle to be captured!
#
# IMPORT All captured frames are copied and buffered in the Sink's processing
# thread. The encoding and saving of each buffered frame is done in the 
# g-idle-thread, therefore, the capture-complete notification is asynchronous.
#
```
<br>

---

* [`encode_and_save_frame_to_jpeg_on_viewer_demand.py`](/examples/python/encode_and_save_frame_to_jpeg_on_viewer_demand.py)
* [`encode_and_save_frame_to_jpeg_on_viewer_demand.cpp`](/examples/cpp/encode_and_save_frame_to_jpeg_on_viewer_demand.cpp)

```python
#
# This example demonstrates the use of a Frame-Capture Sink to encode and
# save video frames to JPEG files on client/viewer demand.
#
# An ODE Frame-Capture Action is provided to The Frame-Capture Sink on creation.
# A client "capture_complete_listener" is added to the the Action to be notified
# when each new file is saved (the ODE Action performs the actual frame-capture).
#
# Child Players (to play the captured image) and Mailers (to mail the image) can
# be added to the ODE Frame-Capture action as well (not shown).
#
# The "invocation" of a new Frame-Capture is done by pressing the "C" key while 
# the Window Sink has user focus... i.e. the xwindow_key_event_handler will call
# the "dsl_sink_frame_capture_initiate" service on key-event.
#
# IMPORT All captured frames are copied and buffered in the Sink's processing
# thread. The encoding and saving of each buffered frame is done in the 
# g-idle-thread, therefore, the capture-complete notification is asynchronous.
#
```
<br>

---

* [`encode_and_save_frame_to_jpeg_thumbnail_periodically.py`](/examples/python/encode_and_save_frame_to_jpeg_thumbnail_periodically.py)
* [`encode_and_save_frame_to_jpeg_thumbnail_periodically.cpp`](/examples/cpp/encode_and_save_frame_to_jpeg_thumbnail_periodically.cpp)

```python
#
# This example demonstrates the use of a Multi-Image Sink to encode and
# save video frames to JPEG files at specified dimensions and frame-rate.
#
# The ouput file path/names are specified using a printf style %d in the 
# provided absolute or relative path. 
#   eample: "./my_images/image.%d04.jpg", will create files in "./my_images/"
#   named "image.0000.jpg", "image.0001.jpg", "image.0002.jpg" etc.
#
# You can limit the number of files that are saved on disc by calling
#   dsl_sink_multi_image_file_max_set. Default = 0 = no max.
#
# Once max-files is reached, old files will be deleted to make room for new
# ones.
#
```



