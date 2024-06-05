# Image Tracking with Inverse Compositional Image Alignment

In this project I implemented Inverse Compositional Image Alignment to track a template across four frames. Here's a breakdown of the key steps:
- **Feature Matching:** I use SIFT (Scale-Invariant Feature Transform) to detect keypoints and compute descriptors in the input images. Then employing a nearest neighbor search to find corresponding points between two images. The ratio test proposed by D. Lowe is used to filter out ambiguous matches.
- **Image Alignment Using Features:**  I aligns two images using the RANSAC (Random Sample Consensus) algorithm to find the best affine transformation matrix. This method iterates over random subsets of correspondences to find the transformation that has the most inliers.

## Result

<table>
  <tr>
    <th>Template</th>
    <th>Target</th>
    <th>SIFT Matches with Ratio Test</th>
  </tr>
  <tr>
    <td><img src="https://github.com/mohitydv09/multi-frame-tracking/assets/101336175/c5c2620a-eeb1-4f6c-88be-d2b4a4fdb1d9" width="180"/></td>
    <td><img src="https://github.com/mohitydv09/multi-frame-tracking/assets/101336175/61b3e340-36b3-4f50-8abd-05c7988dc366" width="300"/></td>
    <td><img src="https://github.com/mohitydv09/multi-frame-tracking/assets/101336175/6fa91f7a-104f-40db-a988-a50abf9bf926" width="500"></td>
  </tr>
</table>

### Affine Transform Calculation Using RANSAC

<p align="center">
<img src="https://github.com/mohitydv09/multi-frame-tracking/assets/101336175/6383d521-ab9e-47d7-8876-657c8a646389" width="500">
</p>

### Image Alignment Steps and Error Map
<p align="center">
<img src="https://github.com/mohitydv09/multi-frame-tracking/assets/101336175/eee9211a-9ee9-4620-8866-bcbb67dd6747" width ="500">
</p>

### Multi-Frame Tracking
<p align="center">
<img src="https://github.com/mohitydv09/multi-frame-tracking/assets/101336175/b9584f95-2bcc-4742-9873-937806081a46" width="500">
</p>
