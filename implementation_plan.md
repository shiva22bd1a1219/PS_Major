# Refining Tumor Segmentation (Skull Stripping Fix)

The segmentation is currently picking up the bright regions of the skull instead of the tumor mass. I will implement a more robust multi-stage filter to isolate the true tumor.

## Proposed Changes

### Backend (`intelliscan backend`)
#### [MODIFY] [run_mode.py](file:///c:/Users/Shiva/Desktop/mlml/PS%21major1/intelliscan/intelliscan%20backend/run_mode.py)
I will upgrade the [predict_segmentation](file:///c:/Users/Shiva/Desktop/mlml/PS%21major1/intelliscan/intelliscan%20backend/run_mode.py#108-178) function with the following filters:
1.  **Center-Priorty Weighting**: I will calculate the distance of each contour from the image center. Contours closer to the edges (like the skull) will be penalized or ignored.
2.  **Solidity Filter**: Tumors are usually solid blobs, while skull arcs are thin. I will filter for contours with high **Solidity** (`Area / ConvexHullArea > 0.6`).
3.  **Intensity Peak Detection**: I will prioritize the contour that contains the absolute highest pixel values in the original image.
4.  **UNet Sanity Check**: I will add a check to see if the UNet predicted a large mass at the edges. If so, I will default back to the refined OpenCV logic.

## Verification Plan
1.  Run `python run_mode.py`.
2.  Test with the specific image reported by the user (the meningioma with visible skull).
3.  Verify the green overlay now correctly snaps to the tumor mass on the right instead of the skull on the left.
