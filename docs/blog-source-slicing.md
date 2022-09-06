# UHD Input Source Slicing with Non-Maximum Processing

```Python
# 4K example image is located under "/deepstream-services-library/test/streams"
image_file = "../../test/streams/4K-image.jpg"

# Create a new streaming image source to stream the UHD file at 10 frames/sec
retval = dsl_source_image_stream_new('image-source', file_path=image_file,
    is_live=False, fps_n=10, fps_d=1, timeout=0)
```

<img src="/Images/0roi_3840x2160_full_frame.png" alt="0 ROIs" width="960">

<table>
  <th>
    <td>Detail of Lower Left Corner</td>
  </th>
  <tr>
    <td>Full image</td>
    <td>1440x1280 ROI No Overlap</td>
    <td>1440x1280 ROI 10% Overlap</td>
  </tr>
  <tr>
    <td>6 objects detected</td>
    <td>8 objects detected</td>
    <td>8 objects detected</td>
  </tr>
  <tr>
    <td><img src="/Images/0roi_3840x2160_full_frame_slice_1.png" alt="0 ROIs Lower Left Corner"></td>
    <td><img src="/Images/3roi_1440x1280_no_overlap_slice_1.png" alt="0 ROIs Lower Center"></td>
    <td><img src="/Images/3roi_1440x1280_no_overlap_slice_1.png" alt="0 ROIs Lower Right Corner"></td>
  </tr>
</table>

</br>
<table>
  <th>
    <td>Detail of Lower Center</td>
  </th>
  <tr>
    <td>Full image</td>
    <td>1440x1280 ROI No Overlap</td>
    <td>1440x1280 ROI 10% Overlap</td>
  </tr>
  <tr>
    <td>1 object detected</td>
    <td>1 object detected</td>
    <td>1 object detected</td>
  </tr>
  <tr>
    <td><img src="/Images/0roi_3840x2160_full_frame_slice_2.png" alt="0 ROIs Lower Left Corner"></td>
    <td><img src="/Images/3roi_1440x1280_no_overlap_slice_2.png" alt="0 ROIs Lower Center"></td>
    <td><img src="/Images/3roi_1440x1280_no_overlap_slice_2.png" alt="0 ROIs Lower Right Corner"></td>
  </tr>
</table>

</br>
<table>
<table>
  <th>
    <td>Detail of Lower Right Corner</td>
  </th>
  <tr>
    <td>Full image</td>
    <td>1440x1280 ROI No Overlap</td>
    <td>1440x1280 ROI 10% Overlap</td>
  </tr>
  <tr>
    <td>3 objects detected, 2 false positives</td>
    <td>4 objects detected, 3 false positives</td>
    <td>4 objects detected, 3 false positives</td>
  </tr>
  <tr>
    <td><img src="/Images/0roi_3840x2160_full_frame_slice_3.png" alt="0 ROIs Lower Left Corner"></td>
    <td><img src="/Images/3roi_1440x1280_no_overlap_slice_3.png" alt="0 ROIs Lower Center"></td>
    <td><img src="/Images/3roi_1440x1280_no_overlap_slice_3.png" alt="0 ROIs Lower Right Corner"></td>
  </tr>
</table>

