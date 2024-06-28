import cv2
import numpy as np
import subprocess

def start_camera_stream():
    command = [
        "libcamera-vid",
        "-t", "0",
        "--inline",
        "-n",
        "--listen",
        "--codec", "mjpeg",
        "--width", "640",
        "--height", "480",
        "-o", "-"
    ]
    return subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)

def main():
    camera_process = start_camera_stream()
    bytes_buffer = b''
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    object_counter = 0  # Counter to keep track of saved images
    
    try:
        while True:
            chunk = camera_process.stdout.read(10240)
            if not chunk:
                break
            bytes_buffer += chunk
            while True:
                start = bytes_buffer.find(b'\xff\xd8')
                end = bytes_buffer.find(b'\xff\xd9', start + 2)
                if start != -1 and end != -1:
                    jpg = bytes_buffer[start:end + 2]
                    bytes_buffer = bytes_buffer[end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    # Apply the background subtractor
                    fg_mask = background_subtractor.apply(frame)
                    # Find contours to detect movements
                    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if cv2.contourArea(contour) > 500:  # Filter small contours
                            x, y, w, h = cv2.boundingRect(contour)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cropped_image = frame[y:y+h, x:x+w]  # Crop the detected object
                            object_counter += 1
                            cv2.imwrite(f'detected_object_{object_counter}.jpg', cropped_image)  # Save the cropped image
                else:
                    break
    finally:
        camera_process.terminate()

if __name__ == '__main__':
    main()
