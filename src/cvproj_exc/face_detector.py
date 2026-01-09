from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from mtcnn import MTCNN


@dataclass
class FaceDetectionResult:
    image: np.ndarray
    """The image."""
    rect: tuple[int, int, int, int]
    """The face bounding box (top left x, top left y, width, height)."""
    aligned: np.ndarray
    """The aligned face image."""


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(
        self, tm_window_size: int = 25, tm_threshold: float = 0.6, aligned_image_size: int = 224 # change
    ) -> None:
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference: Optional[FaceDetectionResult] = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

        # TODO: Specify all parameters for template matching.
        # Template matching parameters
        self.tm_window_size = int(tm_window_size)      # search margin in pixels (+/-)
        self.tm_threshold = float(tm_threshold)        # score threshold
        self.tm_method = cv2.TM_CCOEFF_NORMED          # higher = better, range approx [-1, 1]

        # Tracking state
        self._template_gray: Optional[np.ndarray] = None
        self._last_rect: Optional[tuple[int, int, int, int]] = None

    # TODO: Track a face in a new image using template matching.
    def track_face(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        """
        Track a face in `image` using template matching.
        Only modifies this method: wraps MTCNN calls to avoid crashing when mtcnn throws on empty batches.
        """

        if not hasattr(self, "_lost_count"):
            self._lost_count = 0

        if not hasattr(self, "_lost_patience"):
            self._lost_patience = 3  # 连续失败 3 次才重检

        # Helper: safe detection (do NOT modify detect_face itself)
        def _safe_detect(img: np.ndarray) -> Optional[FaceDetectionResult]:
            try:
                return self.detect_face(img)
            except ValueError:
                # mtcnn can throw when internal batch is empty -> treat as "no face"
                return None

        # Initialize if have no reference/template yet
        if self.reference is None or self._template_gray is None or self._last_rect is None:
            det = _safe_detect(image)
            if det is None:
                # ensure tracker state is reset
                # self.reference = None
                # self._template_gray = None
                # self._last_rect = None
                return None

            self.reference = det
            self._last_rect = det.rect

            tpl = self.crop_face(det.image, det.rect)
            if tpl.size == 0:
                self.reference = None
                self._template_gray = None
                self._last_rect = None
                return None

            self._template_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
            return det

        # Local template matching around last rect
        x, y, w, h = self._last_rect
        if w <= 1 or h <= 1:
            self.reference = None
            self._template_gray = None
            self._last_rect = None
            return None

        img_h, img_w = image.shape[:2]
        margin = self.tm_window_size

        sx0 = max(int(x - margin), 0)
        sy0 = max(int(y - margin), 0)
        sx1 = min(int(x + w + margin), img_w)
        sy1 = min(int(y + h + margin), img_h)

        search_region = image[sy0:sy1, sx0:sx1, :]
        if search_region.size == 0:
            self.reference = None
            self._template_gray = None
            self._last_rect = None
            return None

        search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        tpl = self._template_gray
        th, tw = tpl.shape[:2]

        # If template doesn't fit into search region, try re-detect safely
        if search_gray.shape[0] < th or search_gray.shape[1] < tw:
            det = _safe_detect(image)
            if det is None:
                self.reference = None
                self._template_gray = None
                self._last_rect = None
                return None

            self.reference = det
            self._last_rect = det.rect
            tpl_bgr = self.crop_face(det.image, det.rect)
            if tpl_bgr.size == 0:
                self.reference = None
                self._template_gray = None
                self._last_rect = None
                return None
            self._template_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
            return det

        res = cv2.matchTemplate(search_gray, tpl, self.tm_method)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # Lost track -> maybe re-detect
        if max_val < self.tm_threshold:
            self._lost_count += 1
            if self._lost_count >= self._lost_patience:
                det = _safe_detect(image)
                self._lost_count = 0  # 重检后清零
                if det is None:
                    self.reference = None
                    self._template_gray = None
                    self._last_rect = None
                    return None

                self.reference = det
                self._last_rect = det.rect
                tpl_bgr = self.crop_face(det.image, det.rect)
                if tpl_bgr.size == 0:
                    self.reference = None
                    self._template_gray = None
                    self._last_rect = None
                    return None
                self._template_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
                return det

            # 失败但未到 patience：不重检，不输出
            return None

        # success
        self._lost_count = 0

        best_x = sx0 + int(max_loc[0])
        best_y = sy0 + int(max_loc[1])
        new_rect = (best_x, best_y, int(w), int(h))

        aligned = self.align_face(image, new_rect)
        out = FaceDetectionResult(image=image, rect=new_rect, aligned=aligned)

        # Update state
        self._last_rect = new_rect
        self.reference = out
        tpl_bgr = self.crop_face(image, new_rect)
        if tpl_bgr.size != 0:
            self._template_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)

        return out

    # Face detection in a new image.
    def detect_face(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        # Retrieve all detectable faces in the given image.
        if not (
            detections := self.detector.detect_faces(image, threshold_pnet=0.85, threshold_rnet=0.9)
        ):
            self.reference = None
            return None

        # Select face with the largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]

        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return FaceDetectionResult(rect=face_rect, image=image, aligned=aligned)

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(
            self.crop_face(image, face_rect),
            dsize=(self.aligned_image_size, self.aligned_image_size),
        )

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]
