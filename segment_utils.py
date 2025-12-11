#!/usr/bin/env python3
"""Document Scanner - Automatic Cropping with Rotation Correction"""

import cv2
import numpy as np

class DocumentScanner:
    def __init__(self):
        pass
    
    def order_points(self, pts):
        """Order points: [top-left, top-right, bottom-right, bottom-left]"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    def calculate_rotation_angle(self, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        
        top_angle = np.arctan2(tr[1] - tl[1], tr[0] - tl[0])
        bottom_angle = np.arctan2(br[1] - bl[1], br[0] - bl[0])
        
        return np.degrees((top_angle + bottom_angle) / 2.0)
    
    def rotate_and_crop(self, image, pts):
        """Rotates image based on pts and crops the bounding box."""
        angle = self.calculate_rotation_angle(pts)
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                 flags=cv2.INTER_LINEAR, 
                                 borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=(255, 255, 255))
        
        ones = np.ones(shape=(len(pts), 1))
        pts_ones = np.hstack([pts, ones])
        transformed_pts = M.dot(pts_ones.T).T
        
        x_coords = transformed_pts[:, 0]
        y_coords = transformed_pts[:, 1]
        
        x_min = max(0, int(np.min(x_coords)))
        x_max = min(new_w, int(np.max(x_coords)))
        y_min = max(0, int(np.min(y_coords)))
        y_max = min(new_h, int(np.max(y_coords)))
        
        cropped = rotated[y_min:y_max, x_min:x_max]
        return cropped

    def _detect_canny(self, gray):
        """Canny edge detection for well-lit images with clear contrast."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                return approx.reshape(4, 2)
        
        return None

    def _detect_dark_border(self, gray, image_shape):
        """Detect dark borders around the document."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        height, width = image_shape
        
        for c in contours[:5]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) != 4 or cv2.contourArea(c) < 1000:
                continue
            
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            kernel = np.ones((11, 11), np.uint8) 
            dilated = cv2.dilate(mask, kernel, iterations=1)
            border_ring_mask = cv2.subtract(dilated, mask)
            
            mean_val = cv2.mean(gray, mask=border_ring_mask)[0]
            
            if mean_val < 100: 
                return approx.reshape(4, 2)
                
        return None

    def find_document_contour(self, image, method="canny"):
        """Select detection method."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if method == "dark_border":
            return self._detect_dark_border(gray, gray.shape)
        else:
            return self._detect_canny(gray)

    def process_image(self, image_path, method="canny"):
        """
        Main entry point.
        Args:
            image_path: Path to the image.
            method: 'canny' (default) or 'dark_border'.
        Returns: (processed_image, corner_points)
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read: {image_path}")
            
        h, w = image.shape[:2]
        
        detect_h = 1000
        ratio = detect_h / h if h > detect_h else 1.0
        
        if ratio != 1.0:
            img_resized = cv2.resize(image, (int(w * ratio), detect_h))
        else:
            img_resized = image.copy()

        pts = self.find_document_contour(img_resized, method=method)

        if pts is None:
            print(f"No contour detected ({method}), using original image.")
            processed_img = image
            final_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
        else:
            pts = pts.astype("float32") / ratio
            processed_img = self.rotate_and_crop(image, pts)
            final_pts = pts

        final_pts = self.order_points(final_pts)
        
        return processed_img, final_pts
