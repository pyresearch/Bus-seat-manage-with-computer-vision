import typer
import cv2
import supervision as sv
from ultralytics import YOLO
import pyresearch

# Define the path to the weights file
model = YOLO("last.pt")
app = typer.Typer()

def process_webcam(output_file="output.mp4"):
    cap = cv2.VideoCapture("demo.mp4")  # Replace with 0 for the default webcam

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get the fps of the input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object with 640x640 resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
    out = cv2.VideoWriter(output_file, fourcc, fps, (1080, 1080))

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to 640x640
        frame = cv2.resize(frame, (1080 , 1080))

        # Get detections from the YOLO model
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Annotate the frame
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Write the annotated frame to the output file
        out.write(annotated_frame)

        # Display the frame
        cv2.imshow("Webcam", annotated_frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

@app.command()
def webcam(output_file: str = "output.mp4"):
    typer.echo("Starting webcam processing...")
    process_webcam(output_file)

if __name__ == "__main__":
    app()
