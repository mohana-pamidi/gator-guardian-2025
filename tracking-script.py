import cv2
from collections import defaultdict
from ultralytics import YOLO
import numpy as np
import smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up email server
port = 465
smtp_server = "smtp.gmail.com"
sender_email = "frooia77@gmail.com"
password = input("Type your password and press enter: ")
receiver_email = "mohana.pamidi04@gmail.com"
body = "An employee was spotted not wearing proper protective equipment on site!"
filename = "violator.jpg"

message = MIMEMultipart("alternative")
message["Subject"] = "EMPLOYEE VIOLATING SAFETY RULES"
message["From"] = sender_email
message["To"] = receiver_email
message.attach(MIMEText(body, "plain"))

context = ssl.create_default_context()

# YOLO model training
model = YOLO("best.pt")

cam = cv2.VideoCapture(0)

track_history = defaultdict(lambda: [])

emailSent = False
noDetected = False

while cam.isOpened():
	success, frame = cam.read()
	if(success):
		results = model.track(source=frame, persist=True)

		# Get the boxes and track IDs
		boxes = results[0].boxes.xywh.cpu()
		if(results[0].boxes.id == None):
			continue

		track_ids = results[0].boxes.id.int().cpu().tolist()
		detectedClasses = results[0].boxes.cls.tolist()
		detectedNames = []
		for id in detectedClasses:
			detectedNames.append(model.names[int(id)])
			print("Label: " , model.names[int(id)])
		if not emailSent:
			for class_name in detectedNames:
				if "NO" in class_name:
					noDetected = True
					print("MANAGER NOTIFIED!!!")
			# SEND EMAIL HERE
			if noDetected:
				results[0].save(filename=filename)
				# Find some way to attach image to email
				with open(filename, "rb") as attachment:
					part = MIMEBase("application", "octet-stream")
					part.set_payload(attachment.read())

				encoders.encode_base64(part)
				part.add_header("Content-Disposition", f"attachment; filename= {filename}",)

				message.attach(part)

				with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
					server.login(sender_email, password)
					server.sendmail(sender_email, receiver_email, message.as_string())
				emailSent = True

		# Visualize the results on the frame
		annotated_frame = results[0].plot()

		# Plot the tracks
		for box, track_id in zip(boxes, track_ids):
			x, y, w, h = box
			track = track_history[track_id]
			#print("LABEL:----: ", results[0].names[track_id]) 
			track.append((float(x), float(y)))  # x, y center point
			if len(track) > 30:  # retain 90 tracks for 90 frames
				track.pop(0)

			# Draw the tracking lines
			#points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
			#cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

		# Display the annotated frame
		cv2.imshow("YOLO11 Tracking", annotated_frame)

		# Break the loop if 'q' is pressed
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
	else:
		# Break the loop if the end of the video is reached
		break

# Release the video capture object and close the display window
cam.release()
cv2.destroyAllWindows()
