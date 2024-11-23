import cv2
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
from keras.models import model_from_json
import operator
import sys
from string import ascii_uppercase
from spellchecker import SpellChecker
import os  # To run the main.py file


class Application:
    def __init__(self):
        self.directory = 'model/'
        self.hs = SpellChecker()  # Spell checker instance
        self.vs = cv2.VideoCapture(0)  # Initialize camera
        if not self.vs.isOpened():
            print("Error: Could not access the camera.")
            sys.exit(1)

        self.current_image = None
        self.current_image2 = None

        # Load all models
        self.loaded_model = self.load_model("model-bw")
        self.loaded_model_codpr = self.load_model("model-bw_codpr")
        self.loaded_model_luv = self.load_model("model-bw_luv")
        self.loaded_model_mner = self.load_model("model-bw_mner")
        self.loaded_model_qy = self.load_model("model-bw_qy")

        # Initialize character tracking
        self.ct = {'blank': 0}
        self.blank_flag = 0
        for char in ascii_uppercase:
            self.ct[char] = 0
        print("Loaded models successfully!")

        # Initialize UI
        self.root = tk.Tk()
        self.root.title("Sign Language to Text Converter")
       # self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        # Set the window to full screen
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.bind("<Escape>", self.toggle_full_screen)

        # Add fullscreen background image
        self.add_background_image()

        # UI components initialization
        self.init_ui()

        # Initialize prediction-related variables
        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.frame_count = 0
        self.accumulation_time = 30  # Process every 30 frames

        # Start video loop
        self.video_loop()

    def add_background_image(self):
        """Add a fullscreen background image."""
        background_image_path = "C:/VSCode/project/src/bg.jpeg"  # Path to your background image
        try:
            bg_image = Image.open(background_image_path)
            bg_width, bg_height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            resized_bg = bg_image.resize((bg_width, bg_height), Image.Resampling.LANCZOS)
            bg_photo = ImageTk.PhotoImage(resized_bg)

            bg_label = tk.Label(self.root, image=bg_photo)
            bg_label.image = bg_photo  # Keep a reference to avoid garbage collection
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            print(f"Error loading background image: {e}")

    def load_model(self, model_name):
        """Load a model from JSON and weights files."""
        try:
            with open(f"{self.directory}/{model_name}.json", "r") as json_file:
                model_json = json_file.read()

            model = model_from_json(model_json)
            model.load_weights(f"{self.directory}/best_weights_{model_name.split('-')[-1]}.h5")
            print(f"Model {model_name} successfully loaded!")
            return model
        except FileNotFoundError as fnfe:
            print(f"File not found: {fnfe}. Please ensure the JSON and weights files are in the correct directory.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            sys.exit(1)

    
    def init_ui(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Adjusted camera feed placement
        camera_width = 800
        camera_height = 600
        camera_x = screen_width - camera_width - 50  # Align to the right side
        camera_y = 10  # Top margin for the camera feed

        # Video feed panel (No background color set, transparent over the background)
        self.panel = tk.Label(self.root)
        self.panel.place(x=camera_x, y=camera_y, width=camera_width, height=camera_height)

        # Processed ROI panel (No background color set, transparent over the background)
        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=camera_x + 410, y=camera_y + 65, width=310, height=310)


        # Labels for current character
        self.T1 = tk.Label(self.root, text="Character:", font=("Courier", 20, "bold"), fg="#000000", bg="#FFFFFF")
        self.T1.place(x=800, y=660)

        self.panel3 = tk.Label(self.root, font=("Courier", 25), bg="#FFFFFF", fg="#000000")
        self.panel3.place(x=980, y=660)

        # Labels for current word
        self.T2 = tk.Label(self.root, text="Word:", font=("Courier", 20, "bold"), fg="#000000", bg="#FFFFFF")
        self.T2.place(x=800, y=720)

        self.panel4 = tk.Label(self.root, font=("Courier", 20), bg="#FFFFFF", fg="#000000")
        self.panel4.place(x=980, y=720)

        # Add Stop Camera Button
        stop_button = tk.Button(
            self.root,
            text="Stop Camera",
            command=self.stop_camera,
            font=("Arial", 14, "bold"),
            bg="#4A90E2",
            fg="white",
            relief="flat",
            width=20,
            height=2
        )
        stop_button.place(x=200, y=300)

        # Add Exit Button
        exit_button = tk.Button(
            self.root,
            text="Exit",
            command=self.destructor,
            font=("Arial", 14, "bold"),
            bg="#4A90E2",
            fg="white",
            relief="flat",
            width=20,
            height=2
        )
        exit_button.place(x=200, y=390)

    def stop_camera(self):
        """Stop the camera and redirect to main.py."""
        self.destructor()
        os.system("python main.py")

    def video_loop(self):
        ok, frame = self.vs.read()
        if not ok:
            print("Warning: Unable to read frame from camera.")
            self.root.after(10, self.video_loop)
            return

        self.frame_count += 1

        # Flip and display the video feed (No frame drawn)
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        self.current_image = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=self.current_image)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)

        # Process the selected region of interest (ROI)
        x1 = int(0.5 * frame.shape[1])
        y1 = 10
        x2 = frame.shape[1] - 10
        y2 = int(0.5 * frame.shape[1])
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Predict every 'accumulation_time' frames
        if self.frame_count >= self.accumulation_time:
            self.predict(res)
            self.frame_count = 0

        # Update processed ROI
        self.current_image2 = Image.fromarray(res)
        imgtk = ImageTk.PhotoImage(image=self.current_image2)
        self.panel2.imgtk = imgtk
        self.panel2.config(image=imgtk)

        # Update labels
        self.panel3.config(text=self.current_symbol)
        self.panel4.config(text=self.word)

        self.root.after(30, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))

        # Normalize the image to [0, 1] range (same as during training)
        test_image = test_image.astype('float32') / 255.0

        # Get predictions from the model
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))

         # Validate result size
        if len(result[0]) != 27:  # Expecting 27 outputs (26 letters + 1 blank)
            print(f"Unexpected model output size: {len(result[0])}. Expected 27.")
            self.current_symbol = "Error"
            return

        # Map predictions
        prediction = {'_blank': result[0][0]}  # First index corresponds to "blank"
        for idx, char in enumerate(ascii_uppercase, start=1):
            prediction[char] = result[0][idx]

        # Sort predictions by confidence
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]
        confidence = prediction[0][1]

        # Handle "blank" detection
        if self.current_symbol == "_blank" and confidence > 0.8:  # Ensure high confidence
            self.blank_flag += 1
            if self.blank_flag > 5:  # Consistent blank detection for multiple frames
                if not self.word.endswith(" "):  # Avoid consecutive spaces
                   self.word += " "
                self.blank_flag = 0
        else:
            self.blank_flag = 0
            if self.current_symbol in ascii_uppercase:
                self.word += self.current_symbol


    def destructor(self):
        print("Releasing resources...")
        self.vs.release()
        cv2.destroyAllWindows()
        self.root.quit()

    def toggle_full_screen(self, event=None):
        """Toggle full-screen mode."""
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))
        return "break"


def launch_app():
    app = Application()
    app.root.mainloop()